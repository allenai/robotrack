"""
RoboTracker trainer (torch.distributed, no Lightning), ported from AllTracker's train_stage1.py
(https://github.com/aharley/alltracker, MIT licence) with the same training dynamics:

- Step-based loop over an infinite batch iterator.
- Each GPU belongs to one data group (e.g. 56-frame Kubric, 24-frame RoboTrack-15M), assigned in proportion to the
  groups' ratios; groups differ only in clip shape, so plain DDP works.
- bf16 autocast, AdamW with a one-cycle schedule, gradient clipping.
- Optional gradient accumulation over virtual ranks (distributed.virtual_world_size), so e.g. 4 GPUs reproduce an
  8-GPU run's data mix, batch and update.
- Resumable: checkpoints store each rank's data position and RNG state.
- Optional Weights & Biases logging on rank 0.
"""

import os
import json
import random
import contextlib
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from robotrack.alltracker import data as utils_data
from robotrack.alltracker import misc

from robotrack.training.distributed import (
    init_distributed,
    cleanup,
    is_main_process,
    barrier,
    set_seeds,
)
from robotrack.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    save_final_model,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
)
from robotrack.training.data.composed_dataset import create_dataloader
from robotrack.training.data.infinite_loader import InfiniteBatchIterator
from robotrack.training.forward import ForwardConfig, forward_batch_sparse
from robotrack.training.logging_utils import WandbLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_ranks_to_groups(
    group_names: List[str],
    ratios: List[float],
    world_size: int,
) -> Dict[str, List[int]]:
    """
    Distribute ranks across data groups proportionally to the given ratios.
    Every group gets at least one rank when world_size >= len(groups);
    otherwise extra groups get [].
    """
    assert len(group_names) == len(ratios) and len(group_names) > 0
    ratios = [max(0.0, float(r)) for r in ratios]
    total = sum(ratios)
    assert total > 0, "Sum of group ratios must be > 0"

    if world_size == 1:
        # Single GPU: stick everything in the highest-ratio group.
        idx = max(range(len(ratios)), key=lambda i: ratios[i])
        return {name: ([0] if i == idx else []) for i, name in enumerate(group_names)}

    # Normalize and round, then patch up to land exactly on world_size.
    normed = [r / total for r in ratios]
    counts = [max(1 if world_size >= len(group_names) else 0, int(round(n * world_size)))
              for n in normed]
    diff = world_size - sum(counts)
    # Hand out / reclaim leftover ranks one at a time, biased toward larger groups.
    order = sorted(range(len(group_names)), key=lambda i: -normed[i])
    j = 0
    while diff != 0:
        i = order[j % len(order)]
        if diff > 0:
            counts[i] += 1
            diff -= 1
        elif counts[i] > 1:
            counts[i] -= 1
            diff += 1
        j += 1

    out: Dict[str, List[int]] = {}
    cursor = 0
    for name, c in zip(group_names, counts):
        out[name] = list(range(cursor, cursor + c))
        cursor += c
    return out


def _get_parameter_names(model: nn.Module, forbidden_layer_types) -> List[str]:
    """Names of params not inside any forbidden layer type (used for weight decay grouping)."""
    result: List[str] = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in _get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def _local_checkpoint(path: str) -> str:
    """A local path for a checkpoint path or URL; rank 0 downloads a URL once into torch's cache, the others wait."""
    from robotrack.hub import local_path
    if not path.startswith(("http://", "https://")):
        return path
    if is_main_process():
        local_path(path)        # download once into the cache
    barrier()
    return local_path(path)     # every rank then finds it there


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

        self.rank, self.world_size, self.local_rank, self.device = init_distributed(
            backend=cfg.distributed.backend,
            timeout_mins=cfg.distributed.timeout_mins,
        )

        set_seeds(cfg.seed, rank=self.rank)

        # Optional gradient accumulation over virtual ranks (off unless distributed.virtual_world_size is set):
        # W GPUs emulate an V-GPU run by each hosting V/W virtual ranks and summing their gradients before the
        # optimizer step, so the data mix, per-step batch and update match the V-GPU run.
        vws = cfg.distributed.get("virtual_world_size", None)
        self.virtual_world_size = int(vws) if vws else self.world_size
        if self.virtual_world_size < self.world_size or self.virtual_world_size % self.world_size:
            raise ValueError(f"distributed.virtual_world_size={self.virtual_world_size} must be a multiple of the "
                             f"world size {self.world_size}")
        self.accum_steps = self.virtual_world_size // self.world_size
        self.slots: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cfg.cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cfg.cuda.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.cuda.allow_tf32

        torch.set_float32_matmul_precision("medium")

        # Filled in by setup()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[GradScaler] = None

        # Heterogeneous-rank state. The current rank owns exactly one of these.
        self.my_group: Optional[str] = None
        self.train_loader = None
        self.batch_iterator: Optional[InfiniteBatchIterator] = None
        self.my_inference_iters: int = 4

        # Logging (rank 0)
        self.wandb: Optional[WandbLogger] = None
        self.overpools: Dict[str, misc.SimplePool] = {}

        # Resume state
        self.total_steps = 0
        self.epoch = 0
        self.batches_in_epoch = 0
        self.resume_state: Dict[str, Any] = {}

        self.forward_cfg = ForwardConfig(
            use_huber_loss=bool(cfg.training.get("use_huber_loss", False)),
            visible_loss_weight=float(cfg.training.get("visible_loss_weight", 0.05)),
            invisible_loss_weight=float(cfg.training.get("invisible_loss_weight", 0.01)),
            loss_in_fp32=bool(cfg.training.get("loss_in_fp32", False)),
            clamp_loss_probs=bool(cfg.training.get("clamp_loss_probs", False)),
            skip_nonfinite_outputs=bool(cfg.training.get("skip_nonfinite_batches", False)),
        )

    # -- setup ---------------------------------------------------------------

    def setup(self):
        cfg = self.cfg
        Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)

        if is_main_process():
            with open(Path(cfg.exp_dir) / "config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            with open(Path(cfg.exp_dir) / "meta.json", "w") as f:
                json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

        self._build_model()
        self._build_optimizer()
        self._build_scaler()
        self._assign_groups_and_build_dataloader()
        self._load_checkpoint()
        self._build_batch_iterator()
        self._build_logger()
        self._wrap_ddp()
        barrier()
        self._validate_and_log_setup()
        logging.info(f"Trainer setup complete. Rank {self.rank}/{self.world_size}")

    def _build_model(self):
        logging.info("Building model...")
        self.model = instantiate(self.cfg.model, _recursive_=False)
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {n_params:,}")

    def _build_optimizer(self):
        logging.info("Building optimizer...")
        cfg = self.cfg

        decay = _get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay = [n for n in decay if "bias" not in n]
        groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay],
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay],
                "lr": cfg.optimizer.lr,
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            params=groups,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

        if cfg.scheduler.use_scheduler:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=cfg.optimizer.lr,
                total_steps=cfg.training.num_steps + 100,
                pct_start=cfg.scheduler.pct_start,
                cycle_momentum=False,
                anneal_strategy="cos",
            )
        else:
            self.scheduler = None

    def _build_scaler(self):
        cfg = self.cfg.training
        # bf16 doesn't need loss scaling; only fp16 mixed precision does.
        use_scaler = cfg.mixed_precision and cfg.precision != "bf16"
        self.scaler = GradScaler(enabled=use_scaler)
        logging.info(
            f"GradScaler enabled: {use_scaler} (mixed_precision={cfg.mixed_precision}, precision={cfg.precision})"
        )

    def _assign_groups_and_build_dataloader(self):
        """
        Read cfg.data.groups (ordered list), assign ranks to groups based on
        each group's ``ratio``, then on this rank build only the dataloader
        for this rank's group.

        With distributed.virtual_world_size = V > world_size (gradient accumulation), groups are assigned over V
        virtual ranks and this rank builds one dataloader for each virtual rank it hosts (rank, rank + W, ...), with
        that virtual rank's sampler shard and sample salt; see _train_virtual.
        """
        groups = list(self.cfg.data.groups)
        names = [g.name for g in groups]
        ratios = [float(g.get("ratio", 1.0)) for g in groups]
        rank_map = _assign_ranks_to_groups(names, ratios, self.virtual_world_size)
        if is_main_process():
            logging.info(f"Rank assignment: {rank_map}")
            if self.accum_steps > 1:
                logging.info(f"Virtual ranks: {self.virtual_world_size} over {self.world_size} GPUs, "
                             f"{self.accum_steps} accumulated micro-batches per step")

        self.slots = []
        for k in range(self.accum_steps):
            vrank = self.rank + k * self.world_size
            if self.accum_steps > 1:
                # Python/NumPy RNG as a real rank `vrank` would have it when building its dataset (set_seeds in
                # __init__ seeds with seed + rank); some datasets shuffle their file lists at construction.
                random.seed(self.cfg.seed + vrank)
                np.random.seed(self.cfg.seed + vrank)
            self.slots.append(self._build_group_slot(groups, rank_map, vrank))

        # The first slot is this rank's own (non-virtual) rank: the per-rank attributes below are the ones used by
        # logging, checkpoint metadata and the non-accumulating training loop.
        first = self.slots[0]
        self.my_group = first["group"]
        self.my_inference_iters = first["inference_iters"]
        self.forward_cfg = first["forward_cfg"]
        self.train_loader = first["loader"]

    def _build_group_slot(self, groups, rank_map, vrank: int) -> Dict[str, Any]:
        """Dataloader and forward settings for (virtual) rank ``vrank``; vrank == self.rank without accumulation."""
        # Pick the group this rank belongs to.
        my_group = None
        for name, ranks in rank_map.items():
            if vrank in ranks:
                my_group = name
                break
        assert my_group is not None, (
            f"Rank {vrank} not assigned to any group: {rank_map}"
        )

        # Find the matching group config.
        group_cfg = next(g for g in groups if g.name == my_group)
        inference_iters = int(group_cfg.inference_iters)
        my_batch_size = int(group_cfg.get("batch_size", self.cfg.training.batch_size))
        forward_cfg = self.forward_cfg

        # Each group has independent ranks, so within a group we still want a
        # sampler so its members shard the dataset between them. The legacy
        # Lightning Fabric trainer used global world_size/global_rank here.
        group_ranks = rank_map[my_group]
        group_world_size = len(group_ranks)
        group_local_rank = group_ranks.index(vrank)
        sampler_scope = str(self.cfg.data.get("sampler_scope", "group"))
        if sampler_scope == "global":
            sampler_rank = vrank
            sampler_world_size = self.virtual_world_size
        elif sampler_scope == "group":
            sampler_rank = group_local_rank
            sampler_world_size = group_world_size
        else:
            raise ValueError(f"Unknown data.sampler_scope={sampler_scope!r}; use 'global' or 'group'")
        sampler_seed = int(self.cfg.data.get("sampler_seed", self.cfg.seed))

        dataset = instantiate(group_cfg.dataset, _recursive_=False)
        loader = create_dataloader(
            dataset=dataset,
            batch_size=my_batch_size,
            num_workers=int(group_cfg.get("num_workers", self.cfg.training.num_workers)),
            shuffle=True,
            collate_fn=utils_data.collate_fn_train,
            distributed=(sampler_world_size > 1),
            rank=sampler_rank,
            world_size=sampler_world_size,
            seed=self.cfg.seed,
            sampler_seed=sampler_seed,
            deterministic_samples=bool(self.cfg.training.get("deterministic_samples", True)),
            # sample_salt (default: the group name) keys every sample's randomness; set it to reproduce another run's draws
            seed_salt=f"{group_cfg.get('sample_salt', my_group)}:global{vrank}:local{group_local_rank}:sampler{sampler_scope}",
        )
        logging.info(
            f"Rank {vrank}{f' (on GPU rank {self.rank})' if vrank != self.rank else ''} data group={my_group} "
            f"batch_size={my_batch_size} "
            f"group_local_rank={group_local_rank}/"
            f"{group_world_size} sampler_scope={sampler_scope} sampler_rank={sampler_rank}/"
            f"{sampler_world_size} sampler_seed={sampler_seed}"
        )
        return {"vrank": vrank, "group": my_group, "inference_iters": inference_iters,
                "forward_cfg": forward_cfg, "loader": loader, "iterator": None,
                "epoch": self.epoch, "batches_in_epoch": self.batches_in_epoch}

    def _load_checkpoint(self):
        cfg = self.cfg.checkpoint

        latest = find_latest_checkpoint(cfg.save_dir)
        if latest is not None:
            logging.info(f"Auto-resume from {latest}")
            self._load_training_state(latest)
            return

        if cfg.get("resume_from", None) is not None:
            logging.info(f"Resuming from {cfg.resume_from}")
            self._load_training_state(cfg.resume_from)
            return

        if cfg.get("restore_ckpt", None) is not None:
            path = _local_checkpoint(str(cfg.restore_ckpt))
            logging.info(f"Loading pretrained weights from {path}")
            load_checkpoint(
                path, self.model,
                optimizer=None, scheduler=None, scaler=None,
                strict=cfg.get("strict", False),
            )

    def _load_training_state(self, path: str):
        self.total_steps, self.epoch, self.batches_in_epoch, self.resume_state = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            strict=self.cfg.checkpoint.get("strict", False),
        )
        self._apply_rank_resume_state()

    def _apply_rank_resume_state(self):
        if self.accum_steps > 1:
            return self._apply_virtual_resume_state()
        rank_states = self.resume_state.get("rank_states") if self.resume_state else None
        if rank_states:
            local_state = next(
                (s for s in rank_states if int(s.get("rank", -1)) == self.rank),
                None,
            )
            if local_state is None:
                logging.warning(
                    f"No rank-local resume state for rank {self.rank}; using legacy "
                    f"epoch={self.epoch} batches_in_epoch={self.batches_in_epoch}"
                )
                return

            self.epoch = int(local_state.get("epoch", self.epoch))
            self.batches_in_epoch = int(
                local_state.get("batches_in_epoch", self.batches_in_epoch)
            )
            restore_rng_state(local_state.get("rng_state", None))
            logging.info(
                f"Restored rank-local data state: group={local_state.get('group')} "
                f"epoch={self.epoch} batches_in_epoch={self.batches_in_epoch}"
            )
            return

        # Legacy checkpoints only contain rank-0 RNG state. Avoid applying that
        # to every DDP rank; keep the seeded state for non-rank-0 processes.
        if self.world_size == 1:
            restore_rng_state(self.resume_state.get("rng_states", None))

    def _apply_virtual_resume_state(self):
        """Per-virtual-rank data state. Checkpoints store one rank_states entry per (virtual) rank, so a run can
        resume with or without accumulation as long as the virtual world size is unchanged."""
        rank_states = self.resume_state.get("rank_states") if self.resume_state else None
        by_rank = {int(s.get("rank", -1)): s for s in (rank_states or [])}
        for slot in self.slots:
            state = by_rank.get(slot["vrank"])
            if state is None:
                if rank_states:
                    logging.warning(f"No resume state for virtual rank {slot['vrank']}; using legacy "
                                    f"epoch={self.epoch} batches_in_epoch={self.batches_in_epoch}")
                slot["epoch"], slot["batches_in_epoch"] = self.epoch, self.batches_in_epoch
                continue
            slot["epoch"] = int(state.get("epoch", self.epoch))
            slot["batches_in_epoch"] = int(state.get("batches_in_epoch", self.batches_in_epoch))
            if slot["vrank"] == self.rank:
                restore_rng_state(state.get("rng_state", None))
            logging.info(f"Restored data state of virtual rank {slot['vrank']}: group={state.get('group')} "
                         f"epoch={slot['epoch']} batches_in_epoch={slot['batches_in_epoch']}")
        self.epoch, self.batches_in_epoch = self.slots[0]["epoch"], self.slots[0]["batches_in_epoch"]

    def _build_batch_iterator(self):
        if self.accum_steps > 1:
            for slot in self.slots:
                slot["iterator"] = InfiniteBatchIterator(
                    dataloader=slot["loader"],
                    start_epoch=slot["epoch"],
                    batches_to_skip=slot["batches_in_epoch"],
                    name=f"{slot['group']}:v{slot['vrank']}",
                )
            self.batch_iterator = self.slots[0]["iterator"]
            return
        self.batch_iterator = InfiniteBatchIterator(
            dataloader=self.train_loader,
            start_epoch=self.epoch,
            batches_to_skip=self.batches_in_epoch,
            name=self.my_group,
        )

    def _build_logger(self):
        log_ranks = list(self.cfg.logging.get("log_ranks", [0]))
        if self.rank in log_ranks:
            self._init_pools()

        # WandB only on the main process to avoid duplicate runs.
        if is_main_process() and "wandb" in self.cfg.logging:
            wandb_cfg = self.cfg.logging.wandb
            # On resume, reuse the previous wandb run ID so metrics continue
            # on the same run rather than creating a new one.
            run_id_file = Path(self.cfg.exp_dir) / "wandb_run_id.txt"
            if wandb_cfg.get("id", None) is None and run_id_file.exists():
                saved_id = run_id_file.read_text().strip()
                if saved_id:
                    from omegaconf import open_dict
                    with open_dict(self.cfg):
                        self.cfg.logging.wandb.id = saved_id
                    logging.info(f"Resuming wandb run: {saved_id}")

            self.wandb = WandbLogger(self.cfg.logging.wandb, run_config=self.cfg)

            # Persist the run ID for future resumes.
            if self.wandb.enabled and self.wandb._run is not None:
                run_id_file.write_text(self.wandb._run.id + "\n")

    def _init_pools(self):
        thrs = [1, 2, 4, 8, 16]
        keys = (
            ["d_%d" % t for t in thrs]
            + ["d_avg", "seq_loss_visible", "seq_loss_invisible",
               "vis_loss", "conf_loss", "total_loss"]
        )
        n_pool = self.cfg.logging.get("metric_window", 50)
        min_size = self.cfg.logging.get("metric_min_size", 10)

        # Always create a pool keyed by this rank's dataset name.
        # The dataset itself fills batch.dname[0]; we don't know it until first batch.
        self.overpools = {k: misc.SimplePool(n_pool, version="np", min_size=min_size) for k in keys}

    def _wrap_ddp(self):
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.distributed.find_unused_parameters,
                gradient_as_bucket_view=self.cfg.distributed.gradient_as_bucket_view,
                bucket_cap_mb=self.cfg.distributed.bucket_cap_mb,
                broadcast_buffers=self.cfg.distributed.broadcast_buffers,
            )

    # -- validation log ------------------------------------------------------

    def _validate_and_log_setup(self):
        if not is_main_process():
            return
        cfg = self.cfg
        m = self.model.module if hasattr(self.model, "module") else self.model
        param = next(m.parameters())
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING CONFIGURATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"  exp_name={cfg.exp_name}  exp_dir={cfg.exp_dir}  seed={cfg.seed}")
        logging.info(f"  world_size={self.world_size}  device={self.device}")
        if self.accum_steps > 1:
            logging.info(f"  virtual_world_size={self.virtual_world_size}  accum_steps={self.accum_steps}  "
                         f"rank-0 virtual ranks: {[(sl['vrank'], sl['group']) for sl in self.slots]}")
        logging.info(
            f"  model={type(m).__name__}  params={sum(p.numel() for p in m.parameters()):,} "
            f"trainable={sum(p.numel() for p in m.parameters() if p.requires_grad):,}"
        )
        logging.info(
            f"  precision={cfg.training.precision}  mixed_precision={cfg.training.mixed_precision}  "
            f"scaler_enabled={self.scaler.is_enabled()}"
        )
        logging.info(
            f"  optimizer={type(self.optimizer).__name__}  lr={cfg.optimizer.lr}  "
            f"wd={cfg.optimizer.weight_decay}  scheduler={type(self.scheduler).__name__}"
        )
        logging.info(
            f"  num_steps={cfg.training.num_steps}  batch_size={cfg.training.batch_size}  "
            f"per-rank inference_iters={self.my_inference_iters}  group={self.my_group}"
        )
        logging.info(
            f"  ckpt_dir={cfg.checkpoint.save_dir}  resumed_from_step={self.total_steps}"
        )
        logging.info("=" * 60)

    # -- main loop -----------------------------------------------------------

    def train(self):
        if self.accum_steps > 1:
            return self._train_virtual()
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        scaler = self.scaler
        batch_iter = self.batch_iterator

        amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step in range(self.total_steps, cfg.training.num_steps):
            batch = next(batch_iter)
            utils_data.dataclass_to_cuda_(batch)

            with torch.amp.autocast("cuda",
                                     enabled=cfg.training.mixed_precision,
                                     dtype=amp_dtype):
                total_loss, metrics = forward_batch_sparse(
                    batch, model, self.forward_cfg, None,
                    inference_iters=self.my_inference_iters,
                    sliding_window_len=int(cfg.training.sliding_window_len),
                )

            # Detect loss explosions: a loss several orders of magnitude above
            # normal (~1) indicates a pathological sample or numerical issue.
            # Skipping prevents gradient corruption that cascades into NaN.
            max_loss = cfg.training.get("max_loss_threshold", None)
            if max_loss is not None and total_loss is not None and total_loss.item() > float(max_loss):
                logging.warning(
                    f"[{self.my_group}] loss explosion: {total_loss.item():.1f} > "
                    f"{max_loss} at step {step + 1}, skipping backward"
                )
                total_loss = None

            # Collective skip decision: if ANY rank's batch was invalid, ALL
            # ranks must skip backward, otherwise the surviving ranks deadlock
            # in DDP's all-reduce. (The original train_stage1.py was racy here
            # because lightning's barrier covered it most of the time; we make
            # it explicit.)
            any_skip = torch.tensor(
                [1 if total_loss is None else 0],
                device=self.device, dtype=torch.int32,
            )
            if dist.is_initialized():
                dist.all_reduce(any_skip, op=dist.ReduceOp.SUM)
            if any_skip.item() > 0:
                self.total_steps = step + 1
                if is_main_process():
                    logging.info(
                        f"step {self.total_steps:06d}/{cfg.training.num_steps} skipped "
                        f"({any_skip.item()} rank(s) had invalid batches)"
                    )
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg.training.gradient_clip_norm),
                error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            barrier()  # symmetric with fwd barrier

            self.total_steps = step + 1

            self._maybe_log_step(metrics)
            self._maybe_save_checkpoint()

        self._finish_training()
        cleanup()

    def _forward_slot(self, slot, batch, amp_dtype):
        cfg = self.cfg
        utils_data.dataclass_to_cuda_(batch)
        with torch.amp.autocast("cuda", enabled=cfg.training.mixed_precision, dtype=amp_dtype):
            return forward_batch_sparse(batch, self.model, slot["forward_cfg"], None,
                                        inference_iters=slot["inference_iters"],
                                        sliding_window_len=int(cfg.training.sliding_window_len))

    def _train_virtual(self):
        """
        Training loop with gradient accumulation over virtual ranks (distributed.virtual_world_size = V, W GPUs,
        A = V / W). Each step, every GPU runs its A virtual ranks' micro-batches in turn and backpropagates
        loss / A, with DDP's gradient sync only on the last one. DDP averages over W, so the update uses the mean
        over all V micro-batches, as a V-GPU run would. As in the plain loop, if any micro-batch on any GPU is
        invalid, every GPU skips the whole step.
        """
        cfg = self.cfg
        model, optimizer, scheduler, scaler = self.model, self.optimizer, self.scheduler, self.scaler
        amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
        A = self.accum_steps
        no_sync = model.no_sync if isinstance(model, DDP) else contextlib.nullcontext

        model.train()
        optimizer.zero_grad(set_to_none=True)

        max_loss = cfg.training.get("max_loss_threshold", None)
        for step in range(self.total_steps, cfg.training.num_steps):
            invalid = 0
            first_metrics = None
            for k, slot in enumerate(self.slots):
                batch = next(slot["iterator"])
                last = k == A - 1
                # DDP decides in forward whether the following backward all-reduces, so the non-final micro-batches
                # run forward and backward inside no_sync; the final one syncs the accumulated gradients.
                sync_ctx = contextlib.nullcontext() if last else no_sync()
                with sync_ctx:
                    total_loss, metrics = self._forward_slot(slot, batch, amp_dtype)
                    if not last and total_loss is not None and not (
                            max_loss is not None and total_loss.item() > float(max_loss)):
                        scaler.scale(total_loss / A).backward()
                if max_loss is not None and total_loss is not None and total_loss.item() > float(max_loss):
                    logging.warning(
                        f"[{slot['group']}:v{slot['vrank']}] loss explosion: {total_loss.item():.1f} > "
                        f"{max_loss} at step {step + 1}, skipping backward"
                    )
                    total_loss = None
                if k == 0:
                    first_metrics = metrics
                if total_loss is None:
                    invalid += 1
                if not last:
                    continue

                # Last micro-batch: decide collectively before the synchronising backward.
                any_skip = torch.tensor([invalid], device=self.device, dtype=torch.int32)
                if dist.is_initialized():
                    dist.all_reduce(any_skip, op=dist.ReduceOp.SUM)
                if any_skip.item() > 0:
                    optimizer.zero_grad(set_to_none=True)
                    break
                scaler.scale(total_loss / A).backward()
            else:
                any_skip = None

            if any_skip is not None:
                self.total_steps = step + 1
                if is_main_process():
                    logging.info(
                        f"step {self.total_steps:06d}/{cfg.training.num_steps} skipped "
                        f"({any_skip.item()} invalid micro-batch(es))"
                    )
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg.training.gradient_clip_norm),
                error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            barrier()

            self.total_steps = step + 1

            self._maybe_log_step(first_metrics)
            self._maybe_save_checkpoint()

        self._finish_training()
        cleanup()

    # -- periodic actions ----------------------------------------------------

    def _maybe_log_step(self, metrics):
        log_ranks = list(self.cfg.logging.get("log_ranks", [0]))
        if self.rank not in log_ranks or metrics is None:
            return

        scalar_freq = min(int(self.cfg.logging.get("scalar_freq", 53)),
                          int(self.cfg.logging.log_freq))
        save_scalar = (self.total_steps % scalar_freq == 0)

        if not save_scalar:
            self._print_progress(metrics)
            return

        for key in self.overpools.keys():
            if key in metrics:
                self.overpools[key].update([metrics[key]])

        if self.wandb is not None and self.wandb.enabled:
            wb_metrics = {
                "lr": self.optimizer.param_groups[-1]["lr"],
                f"train/{self.my_group}/total_loss": metrics["total_loss"],
            }
            for key in self.overpools.keys():
                if self.overpools[key].have_min_size():
                    wb_metrics[f"train/{self.my_group}/{key}"] = self.overpools[key].mean()
            self.wandb.log_dict(wb_metrics, step=self.total_steps)

        self._print_progress(metrics)

    def _print_progress(self, metrics):
        loss_str = ""
        if metrics is not None and "total_loss" in metrics:
            loss_str = f" loss={metrics['total_loss']:.3f}"
        if "d_avg" in self.overpools and self.overpools["d_avg"].have_min_size():
            loss_str += f" d_avg={self.overpools['d_avg'].mean()*100:.1f}"
        logging.info(
            f"[{self.my_group}] step {self.total_steps:06d}/"
            f"{self.cfg.training.num_steps}{loss_str}  (rank {self.rank})"
        )

    def _maybe_save_checkpoint(self):
        save_every = self.cfg.checkpoint.save_every_n_steps
        if not save_every or self.total_steps % save_every != 0:
            return
        self._save_step_checkpoint()
        barrier()

    def _gather_rank_states(self) -> Optional[List[Dict[str, Any]]]:
        if self.accum_steps > 1:
            rng = capture_rng_state()
            local = []
            for slot in self.slots:
                it = slot["iterator"].state_dict()
                local.append({"rank": slot["vrank"], "gpu_rank": self.rank, "local_rank": self.local_rank,
                              "group": slot["group"], "epoch": it["epoch"],
                              "batches_in_epoch": it["batches_in_epoch"],
                              "rng_state": rng if slot["vrank"] == self.rank else None})
            if dist.is_initialized():
                gathered = [None for _ in range(self.world_size)] if is_main_process() else None
                dist.gather_object(local, gathered, dst=0)
                if not is_main_process():
                    return None
                return sorted((s for part in gathered for s in part), key=lambda s: s["rank"])
            return local
        iter_state = self.batch_iterator.state_dict()
        local_state = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "group": self.my_group,
            "epoch": iter_state["epoch"],
            "batches_in_epoch": iter_state["batches_in_epoch"],
            "rng_state": capture_rng_state(),
        }

        if dist.is_initialized():
            gathered = [None for _ in range(self.world_size)] if is_main_process() else None
            dist.gather_object(local_state, gathered, dst=0)
            return gathered if is_main_process() else None

        return [local_state]

    def _save_step_checkpoint(self):
        save_path = Path(self.cfg.checkpoint.save_dir) / f"step_{self.total_steps:06d}.pth"
        iter_state = self.batch_iterator.state_dict()
        rank_states = self._gather_rank_states()
        save_checkpoint(
            str(save_path),
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.total_steps,
            iter_state["epoch"],
            self.cfg,
            rank=self.rank,
            batches_in_epoch=iter_state["batches_in_epoch"],
            rank_states=rank_states,
        )
        self._update_latest_checkpoint_alias(save_path)
        cleanup_old_checkpoints(
            self.cfg.checkpoint.save_dir,
            self.cfg.checkpoint.get("num_checkpoints", None),
            rank=self.rank,
        )

    def _update_latest_checkpoint_alias(self, save_path: Path):
        if self.rank != 0:
            return

        latest_path = Path(self.cfg.checkpoint.save_dir) / "latest.pth"
        tmp_path = latest_path.with_suffix(".pth.tmp")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            os.link(save_path, tmp_path)
            os.replace(tmp_path, latest_path)
            logging.info(f"Updated latest checkpoint alias: {latest_path} -> {save_path.name}")
        except OSError as e:
            logging.warning(
                f"Could not hard-link latest checkpoint alias ({e}); copying instead."
            )
            if tmp_path.exists():
                tmp_path.unlink()
            shutil.copy2(save_path, tmp_path)
            os.replace(tmp_path, latest_path)
            logging.info(f"Updated latest checkpoint copy: {latest_path}")

    def _finish_training(self):
        ckpt_path = Path(self.cfg.checkpoint.save_dir) / f"step_{self.total_steps:06d}.pth"
        final_path = Path(self.cfg.checkpoint.save_dir) / "final_model.pth"
        if is_main_process():
            save_final_model(str(final_path), self.model, rank=self.rank)

        barrier()

        needs_checkpoint = not ckpt_path.exists() if is_main_process() else False
        if dist.is_initialized():
            needs_tensor = torch.tensor(
                [1 if needs_checkpoint else 0],
                device=self.device,
                dtype=torch.int32,
            )
            dist.broadcast(needs_tensor, src=0)
            needs_checkpoint = bool(needs_tensor.item())

        if needs_checkpoint:
            self._save_step_checkpoint()
            barrier()

        if is_main_process() and self.wandb is not None:
            self.wandb.finish()
        if is_main_process():
            logging.info("Training finished.")


def create_trainer(cfg: DictConfig) -> Trainer:
    return Trainer(cfg)
