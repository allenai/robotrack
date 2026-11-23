"""
Checkpoint utilities with backward compatibility for legacy AllTracker checkpoints.

Supports loading:
1. Plain state_dict
2. {'model': ..., 'optimizer': ...} dict format (alltracker original format)
3. Keys with 'module.' prefix (DDP wrapped)
4. Keys with '_orig_mod.' prefix (torch.compile wrapper)
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip DDP and torch.compile prefixes."""
    if not state_dict:
        return state_dict
    normalized: Dict[str, Any] = {}
    stripped_compile = False
    for k, v in state_dict.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module.") :]
        while nk.startswith("_orig_mod."):
            stripped_compile = True
            nk = nk[len("_orig_mod.") :]
        normalized[nk] = v
    if stripped_compile:
        logging.info("Stripped '_orig_mod.' prefix from checkpoint keys (torch.compile)")
    return normalized


def _unwrap_for_state_dict(model: nn.Module) -> nn.Module:
    """DDP -> inner module; torch.compile -> original module."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def capture_rng_state() -> Dict[str, Any]:
    """Capture this process's RNG state."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    """Restore this process's RNG state if present."""
    if not rng_state:
        return
    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state and rng_state["torch"] is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "cuda" in rng_state and rng_state["cuda"] is not None:
        cuda_state = rng_state["cuda"]
        if isinstance(cuda_state, (list, tuple)):
            local_index = torch.cuda.current_device()
            if local_index < len(cuda_state):
                cuda_state = cuda_state[local_index]
            else:
                cuda_state = cuda_state[0]
        torch.cuda.set_rng_state(cuda_state)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = False,
    filter_keys: Optional[List[str]] = None,
) -> Tuple[int, int, int, Optional[Dict]]:
    """
    Load checkpoint with legacy format support.

    Returns (total_steps, epoch, batches_in_epoch, rng_states).
    """
    if filter_keys is None:
        # alltracker registers a 'time_emb' buffer whose shape can change with seqlen;
        # filter it so different-length runs don't clash.
        filter_keys = ["time_emb"]

    logging.info(f"Loading checkpoint from {path}")
    state_dict = torch.load(path, weights_only=False, map_location="cpu")

    if "model" in state_dict:
        model_state = state_dict["model"]
    else:
        model_state = state_dict

    model_state = _normalize_state_dict_keys(model_state)

    if filter_keys:
        original_len = len(model_state)
        model_state = {
            k: v for k, v in model_state.items()
            if not any(fk in k for fk in filter_keys)
        }
        if len(model_state) < original_len:
            logging.info(
                f"Filtered out {original_len - len(model_state)} keys containing {filter_keys}"
            )

    target = _unwrap_for_state_dict(model)
    missing, unexpected = target.load_state_dict(model_state, strict=strict)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    total_steps = 0
    if optimizer is not None and "optimizer" in state_dict:
        logging.info("Loading optimizer state")
        try:
            optimizer.load_state_dict(state_dict["optimizer"])
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")

    if scheduler is not None and "scheduler" in state_dict:
        logging.info("Loading scheduler state")
        try:
            scheduler.load_state_dict(state_dict["scheduler"])
        except Exception as e:
            logging.warning(f"Failed to load scheduler state: {e}")

    if scaler is not None and "scaler" in state_dict:
        logging.info("Loading scaler state")
        try:
            scaler.load_state_dict(state_dict["scaler"])
        except Exception as e:
            logging.warning(f"Failed to load scaler state: {e}")

    # Accept both new "total_steps" and legacy "global_step" keys.
    if "total_steps" in state_dict:
        total_steps = int(state_dict["total_steps"])
    elif "global_step" in state_dict:
        total_steps = int(state_dict["global_step"])
    if total_steps:
        logging.info(f"Resuming from step {total_steps}")

    epoch = int(state_dict.get("epoch", 0))
    batches_in_epoch = int(state_dict.get("batches_in_epoch", 0))
    resume_state = {
        "rng_states": state_dict.get("rng_states", None),
        "rank_states": state_dict.get("rank_states", None),
    }

    logging.info("Checkpoint loaded successfully")
    return total_steps, epoch, batches_in_epoch, resume_state


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    total_steps: int,
    epoch: int,
    config: Optional[DictConfig] = None,
    rank: int = 0,
    batches_in_epoch: int = 0,
    rank_states: Optional[List[Dict[str, Any]]] = None,
):
    """Save full training state (only on rank 0)."""
    if rank != 0:
        return

    model_to_save = _unwrap_for_state_dict(model)
    rng_states = capture_rng_state()

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "total_steps": total_steps,
        "global_step": total_steps,  # legacy alias
        "epoch": epoch,
        "batches_in_epoch": batches_in_epoch,
        "rng_states": rng_states,
        "rank_states": rank_states,
    }

    if config is not None:
        state["config"] = OmegaConf.to_container(config, resolve=True)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    logging.info(f"Saved checkpoint to {path}")


def _extract_step(filename: str) -> int:
    """Pull the step number out of a checkpoint filename."""
    # step_XXXXXX.pth (new format)
    m = re.search(r"step_(\d+)\.pth", filename)
    if m:
        return int(m.group(1))
    # model-XXXXXXXXX.pth (legacy alltracker format)
    m = re.search(r"model-(\d+)\.pth", filename)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", filename)
    if nums:
        return int(nums[-1])
    return 0


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the highest-step checkpoint in ckpt_dir, or None."""
    if not os.path.exists(ckpt_dir):
        return None
    latest_alias = os.path.join(ckpt_dir, "latest.pth")
    if os.path.exists(latest_alias):
        logging.info(f"Found latest checkpoint alias: {latest_alias}")
        return latest_alias
    files = [
        f for f in os.listdir(ckpt_dir)
        if (
            f.endswith(".pth")
            and "final" not in f
            and f != "latest.pth"
            and not os.path.isdir(os.path.join(ckpt_dir, f))
        )
    ]
    if not files:
        return None
    latest = max(files, key=_extract_step)
    logging.info(f"Found latest checkpoint: {latest} (step {_extract_step(latest)})")
    return os.path.join(ckpt_dir, latest)


def cleanup_old_checkpoints(ckpt_dir: str, num_to_keep: Optional[int], rank: int = 0):
    """Keep only the N newest step_*.pth in ckpt_dir."""
    if rank != 0 or num_to_keep is None:
        return
    if not os.path.exists(ckpt_dir):
        return
    checkpoints = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth") and "final" not in f and f != "latest.pth":
            step = _extract_step(f)
            checkpoints.append((step, os.path.join(ckpt_dir, f)))
    checkpoints.sort(reverse=True)
    num_to_keep = max(num_to_keep, 1)
    for step, path in checkpoints[num_to_keep:]:
        logging.info(f"Deleting old checkpoint: {path}")
        try:
            os.remove(path)
        except OSError as e:
            logging.warning(f"Failed to delete {path}: {e}")


def save_final_model(path: str, model: nn.Module, rank: int = 0):
    """Weights-only export for inference."""
    if rank != 0:
        return
    model_to_save = _unwrap_for_state_dict(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_to_save.state_dict()}, path)
    logging.info(f"Saved final model to {path}")
