"""Train RoboTracker-Warp: fine-tune CoWTracker on 50% Kubric + 50% RoboTrack-15M.

    git clone --recursive https://github.com/facebookresearch/cowtracker /path/to/cowtracker
    torchrun --nproc_per_node=8 -m robotrack.cowtracker.train --cowtracker-repo /path/to/cowtracker \
        --kubric-root data/alltracker --robotrack15m-root data/RoboTrack-15M --exp robotracker_warp

LICENSE: this folder is under the FAIR Noncommercial Research License (LICENSE here), not the repository's Apache 2.0:
it runs on CoWTracker's code, and train_head_forward() below follows CoWTracker's CowTrackingHead.forward. The weights
it starts from and produces are CC-BY-NC 4.0. Research and non-commercial use only.

CoWTracker releases no training code. The recipe follows the CoWTracker paper (arXiv 2602.04877, Sec. 4.1) and, for the
losses, CoTracker3, which the paper says it follows:
  - 16-frame clips at 336x560, pixel values 0..255, one clip per GPU on 8 GPUs, 10k steps.
  - VGGT's patch embedding is frozen (paper). So is the branch that is inactive in the released model: VGGT layers
    18-23 feed a DPT branch whose output the following ReLU always zeroes, so they get no gradient; freezing them
    keeps weight decay from moving them. The output is unchanged.
  - Loss on every refinement pass, weighted 0.8^(K-1-i): Huber (delta 6) on visible points x0.05, L1 on occluded
    points x0.01, visibility BCE, and confidence BCE (target: within 12 px) on visible points.
  - Every training point is queried at frame 0 (the model tracks the first frame's pixels); its prediction is read
    from the dense output at its frame-0 position.
  - AdamW (lr 1e-5, weight decay 1e-4), linear warm-up over 5% of the steps then cosine decay, bf16 autocast,
    gradient clipping at 1.0.
Data: the groups of robotrack/training/config/data/robotrack15m_50_mix.yaml (the RoboTracker mix) at this clip size;
each sample's group is drawn by the group ratios, and its randomness is keyed by (seed, rank, epoch, index, group).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

DATA_CONFIG = Path(__file__).resolve().parents[1] / "training/config/data/robotrack15m_50_mix.yaml"

# The inactive branch of the released model (see above): VGGT blocks 18-23 and the DPT layers of that branch.
DEAD = tuple(f"aggregator.{k}_blocks.{i}." for k in ("frame", "global") for i in range(18, 24)) + tuple(
    f"feature_extractor.dpt_head.{m}." for m in ("projects.3", "resize_layers.3", "scratch.layer4_rn"))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cowtracker-repo", required=True, help="clone of facebookresearch/cowtracker (with submodules)")
    p.add_argument("--kubric-root", required=True, help="folder with AllTracker's kubric_au/ and ce64/kublong/")
    p.add_argument("--robotrack15m-root", required=True, help="RoboTrack-15M as downloaded from Hugging Face")
    p.add_argument("--exp", default="robotracker_warp")
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/<exp>")
    p.add_argument("--restore", default=None, help="start weights; default: the released CoWTracker (downloaded)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wdecay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--batch-per-gpu", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--crop", type=int, nargs=2, default=[336, 560])
    p.add_argument("--traj-per-sample", type=int, default=256)
    p.add_argument("--ckpt-head", action="store_true", help="gradient-checkpoint each refinement pass (less memory)")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- data
def build_groups(args, workers_total):
    import hydra.utils
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"kubric_root": args.kubric_root, "robotrack15m_root": args.robotrack15m_root,
                            "data": OmegaConf.load(DATA_CONFIG)})
    OmegaConf.resolve(cfg)
    groups = cfg.data.groups
    workers = max(1, (workers_total - 1) // len(groups))
    out = []
    for g in groups:
        d = g.dataset
        d.crop_size, d.seq_len, d.traj_per_sample = list(args.crop), args.seq_len, args.traj_per_sample
        # Loaders shuffle their file lists at construction with the global RNG; seed it so every rank and every
        # rebuild maps indices to the same clips.
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        out.append({"name": g.name, "salt": g.get("sample_salt", g.name), "ratio": float(g.ratio),
                    "ds": hydra.utils.instantiate(d), "workers": workers})
    return out


def infinite(ds, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    epoch = 0
    while True:
        ds.set_epoch(epoch)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True, num_workers=workers, generator=g,
                                         persistent_workers=False, collate_fn=lambda x: x)
        for sample, gotit in dl:
            if gotit:
                yield sample
        epoch += 1


def make_batches(groups, batch, rank, seed):
    from robotrack.training.data.deterministic_dataset import DeterministicSampleDataset
    rng = random.Random(seed * 1000 + rank)
    its = [infinite(DeterministicSampleDataset(g["ds"], base_seed=seed * 1000 + rank, salt=g["salt"]), g["workers"],
                    seed * 1000 + rank * 10 + i) for i, g in enumerate(groups)]
    w = [g["ratio"] for g in groups]
    while True:
        samples, names = [], []
        for _ in range(batch):
            k = rng.choices(range(len(groups)), weights=w)[0]
            samples.append(next(its[k]))
            names.append(groups[k]["name"])
        # Every track the loader returns is supervised; clips are padded to the batch's largest count with invalid points.
        nmax = max(s.trajs.shape[1] for s in samples)
        vids, trajs, vis, val = [], [], [], []
        for s in samples:
            pad = nmax - s.trajs.shape[1]
            vids.append(s.video)
            trajs.append(F.pad(s.trajs, (0, 0, 0, pad)))
            vis.append(F.pad(s.visibs.float(), (0, pad)))
            val.append(F.pad(s.valids.float(), (0, pad)))
        yield torch.stack(vids), torch.stack(trajs), torch.stack(vis), torch.stack(val), names


# ---------------------------------------------------------------- model
def load_model(path):
    from cowtracker.models.cowtracker import CoWTracker
    model = CoWTracker()
    sd = CoWTracker._load_checkpoint(path)    # None: CoWTracker's loader downloads the released weights
    sd = sd.get("model", sd)
    if any(k.startswith("tracking_head.aggregator.") for k in sd):
        sd = CoWTracker._remap_legacy_state_dict(sd)
    return model, model.load_state_dict(sd, strict=False)


def frozen_by_recipe(name):
    """VGGT's patch embedding (paper) and the inactive layer-23 branch (above)."""
    # refinenet4.resConfUnit2.conv1 sees only the zeroed branch: its weight gets no gradient (its bias still does).
    return (name.startswith("aggregator.patch_embed.") or name.startswith(DEAD)
            or name == "feature_extractor.dpt_head.scratch.refinenet4.resConfUnit2.conv1.weight")


def train_head_forward(head, features, image_size, ckpt):
    """CoWTracker's CowTrackingHead.forward, returning every refinement pass: list of (tracks, vis_logit, conf_logit).
    Same modules and weights; each pass is upsampled with its own hidden state, as RAFT-style trainers do."""
    import cowtracker.heads.tracking_head as th
    B, S, _, H, W = features.shape
    H_img, W_img = image_size
    fmap = head.fmap_conv(features.view(B * S, -1, H, W)).view(B, S, -1, H, W)
    frame0 = fmap[:, 0:1].expand(B, S, -1, H, W)
    net = head.hidden_conv(torch.cat([frame0, fmap], dim=2).view(B * S, -1, H, W)).view(B, S, -1, H, W)
    flow = torch.zeros(B, S, 2, H, W, device=features.device, dtype=features.dtype)
    coords = th.coords_grid(B * S, H, W, device=features.device).to(fmap.dtype).view(B, S, 2, H, W)

    def one_pass(flow, net):
        warped = th.bilinear_sampler(fmap.reshape(B * S, -1, H, W),
                                     (coords + flow).view(B * S, 2, H, W).permute(0, 2, 3, 1)).view(B, S, -1, H, W)
        inp = head.warp_linear(torch.cat([frame0, warped, net, flow], dim=2).view(B * S, -1, H, W)).view(B, S, -1, H, W)
        # CoWTracker's video transformer supports one clip at a time; clips do not interact in it.
        out = torch.cat([head.refine_net(inp[b:b + 1])["out"] for b in range(B)], 0)
        net = head.refine_transform(torch.cat([out.view(B * S, -1, H, W), net.view(B * S, -1, H, W)], dim=1)
                                    ).view(B, S, -1, H, W)
        upd = head.flow_head(net.view(B * S, -1, H, W)).view(B, S, 4, H, W)
        return flow + upd[:, :, :2], upd[:, :, 2:], net

    preds = []
    for _ in range(head.warp_iters):
        flow = flow.detach()
        if ckpt:
            flow, info, net = torch.utils.checkpoint.checkpoint(one_pass, flow, net, use_reentrant=False)
        else:
            flow, info, net = one_pass(flow, net)
        weight = 0.25 * head.upsample_weight(net.view(B * S, -1, H, W)).view(B, S, -1, H, W)
        flow_up, info_up = head._upsample_predictions(flow, info, weight)
        preds.append((head._flow_to_tracks(flow_up, H_img, W_img), info_up[..., 0], info_up[..., 1]))
    return preds


def forward_train(model, video, ckpt_head):
    images = video / 255.0
    tokens, patch_idx = model.aggregator(images)
    features = model.feature_extractor(tokens, images, patch_idx)
    return train_head_forward(model.tracking_head, features, images.shape[-2:], ckpt_head)


def sample_at(field, q, H, W):
    """field B,S,H,W[,C] -> values at frame-0 query positions q (B,N,2, pixels): B,S,N[,C]."""
    squeeze = field.ndim == 4
    if squeeze:
        field = field.unsqueeze(-1)
    B, S, _, _, C = field.shape
    grid = torch.stack([2 * q[..., 0] / (W - 1) - 1, 2 * q[..., 1] / (H - 1) - 1], -1)
    grid = grid[:, None].expand(B, S, -1, -1).reshape(B * S, 1, -1, 2).to(field.dtype)
    v = F.grid_sample(field.permute(0, 1, 4, 2, 3).reshape(B * S, C, H, W), grid, align_corners=True)
    v = v.view(B, S, C, -1).permute(0, 1, 3, 2)
    return v[..., 0] if squeeze else v


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


def compute_loss(preds, trajs, vis, valid, H, W, gamma=0.8):
    q = trajs[:, 0]
    K = len(preds)
    track_vis = track_occ = vis_bce = conf_bce = 0.0
    for i, (tr, vl, cl) in enumerate(preds):
        w = gamma ** (K - i - 1)
        p = sample_at(tr.float(), q, H, W)          # B,S,N,2
        v = sample_at(vl.float(), q, H, W)
        c = sample_at(cl.float(), q, H, W)
        diff = p - trajs
        ad = diff.abs()
        huber = torch.where(ad <= 6.0, 0.5 * diff ** 2, 6.0 * (ad - 3.0)).mean(-1)
        track_vis = track_vis + w * masked_mean(huber, valid * vis)
        track_occ = track_occ + w * masked_mean(ad.mean(-1), valid * (1 - vis))
        vis_bce = vis_bce + masked_mean(F.binary_cross_entropy_with_logits(v, vis, reduction="none"), valid)
        err = ((p.detach() - trajs) ** 2).sum(-1)
        conf_bce = conf_bce + masked_mean(
            F.binary_cross_entropy_with_logits(c, (err <= 144.0).float(), reduction="none"), valid * vis)
    parts = {"track_vis": 0.05 * track_vis / K, "track_occ": 0.01 * track_occ / K, "vis": vis_bce / K,
             "conf": conf_bce / K}
    return sum(parts.values()), parts


def save_atomic(obj, path):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main
def main():
    a = parse()
    sys.path.insert(0, str(Path(a.cowtracker_repo).resolve()))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    cpus = len(os.sched_getaffinity(0)) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    out = a.out_dir or Path("runs") / a.exp
    ckdir = out / "checkpoints"
    if rank == 0:
        ckdir.mkdir(parents=True, exist_ok=True)
        json.dump({**{k: str(v) for k, v in vars(a).items()}, "world_size": world, "argv": sys.argv,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out / "launch_record.json", "w"), indent=1)

    model, msg = load_model(a.restore)
    assert not msg.missing_keys, f"start weights miss {msg.missing_keys[:10]}"
    for n, p in model.named_parameters():
        p.requires_grad = not frozen_by_recipe(n)
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, "
              f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wdecay)
    warm = max(1, int(a.warmup_frac * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    step, latest = 0, ckdir / "latest.pth"
    if latest.exists():
        st = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"])
        step = st["step"]
        if rank == 0:
            print(f"resumed from {latest} at step {step}", flush=True)

    groups = build_groups(a, cpus)
    if rank == 0:
        print("groups:", [(g["name"], g["ratio"], len(g["ds"])) for g in groups], flush=True)
    batches = make_batches(groups, a.batch_per_gpu, rank, a.seed + step)
    H, W = a.crop
    wb = None
    if a.wandb and rank == 0:
        import wandb
        wb = wandb.init(project="robotrack", name=a.exp, config=vars(a), resume="allow")
    t0 = time.time()
    while step < a.steps:
        opt.zero_grad(set_to_none=True)
        video, trajs, vis, valid, names = (x.to(dev, non_blocking=True) if torch.is_tensor(x) else x for x in next(batches))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = forward_train(model, video, a.ckpt_head)
        loss, parts = compute_loss(preds, trajs, vis, valid, H, W)
        loss.backward()
        if ddp:   # every rank starts from the same weights: averaging the gradients keeps them identical
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if rank == 0 and (step % a.log_every == 0 or step == 1):
            print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v.item():.4f}" for k, v in parts.items())
                  + f" gnorm {gn.item():.2f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s {names}", flush=True)
            if wb:
                wb.log({"loss": loss.item(), **{k: v.item() for k, v in parts.items()}, "lr": sched.get_last_lr()[0]},
                       step=step)
        if rank == 0 and (step % a.save_every == 0 or step == a.steps):
            # {"model": ...} is the format of the released checkpoints: robotrack.eval_cowtracker loads it directly
            save_atomic({"model": model.state_dict(), "step": step}, ckdir / f"step_{step:06d}.pth")
            save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                         "step": step}, latest)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
