"""
Training forward pass and loss for point tracks, ported from AllTracker's train_stage1.py::forward_batch_sparse
(https://github.com/aharley/alltracker, MIT licence). Losses, sliding-window slicing and scalar metrics follow the
original.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import contextlib

import numpy as np
import torch

from robotrack.alltracker import basic, loss, samp


@dataclass
class ForwardConfig:
    """Subset of training config that the forward pass needs."""
    use_huber_loss: bool = False
    visible_loss_weight: float = 0.05
    invisible_loss_weight: float = 0.01
    loss_in_fp32: bool = False
    clamp_loss_probs: bool = False
    skip_nonfinite_outputs: bool = False


def forward_batch_sparse(
    batch,
    model: torch.nn.Module,
    cfg: ForwardConfig,
    sw,  # unused, kept for API compatibility
    inference_iters: int,
    sliding_window_len: int = 16,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Compute Stage-1 (Kubric) total_loss and metrics for one micro-batch.

    Returns (total_loss, metrics) or (None, None) if the sample is unusable.
    """
    rgbs = batch.video
    trajs_g = batch.trajs
    vis_g = batch.visibs       # B,S,N
    valids = batch.valids
    dname = batch.dname

    B, T, C, H, W = rgbs.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = rgbs.device
    S = sliding_window_len

    if N == 0 or torch.any(vis_g[:, 0].reshape(-1) == 0):
        # First frame must have all queries visible — original behavior.
        print("vis_g", vis_g.shape)
        return None, None

    all_flow_e, all_visconf_e, all_flow_preds, all_visconf_preds = model(
        rgbs, iters=inference_iters, sw=None, is_training=True
    )

    if cfg.skip_nonfinite_outputs and (
        torch.isnan(all_flow_e).any() or torch.isinf(all_flow_e).any()
    ):
        print(f"[forward_batch_sparse] NaN/Inf in model flow output, skipping batch")
        return None, None

    grid_xy = basic.gridcloud2d(1, H, W, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
    traj_maps_e = all_flow_e + grid_xy  # B,T,2,H,W

    xy0 = trajs_g[:, 0]  # B,N,2
    xy0[:, :, 0] = xy0[:, :, 0].clamp(0, W - 1)
    xy0[:, :, 1] = xy0[:, :, 1].clamp(0, H - 1)

    traj_maps_e_ = traj_maps_e.reshape(B * T, 2, H, W)
    xy0_full = xy0.reshape(B, 1, N, 2).repeat(1, T, 1, 1).reshape(B * T, N, 1, 2)
    trajs_e_ = samp.bilinear_sampler(traj_maps_e_, xy0_full)
    trajs_e = trajs_e_.reshape(B, T, 2, N).permute(0, 1, 3, 2)  # B,T,N,2

    # Per-window predicted trajectories (for sequence loss)
    xy0_win = xy0.reshape(B, 1, N, 2).repeat(1, S, 1, 1).reshape(B * S, N, 1, 2)
    coord_predictions = []
    assert B == 1
    for fpl in all_flow_preds:
        cps = []
        for fp in fpl:
            traj_map = fp + grid_xy           # B,S,2,H,W
            traj_map_ = traj_map.reshape(B * S, 2, H, W)
            traj_e_ = samp.bilinear_sampler(traj_map_, xy0_win)
            traj_e = traj_e_.reshape(B, S, 2, N).permute(0, 1, 3, 2)
            cps.append(traj_e)
        coord_predictions.append(cps)

    # Visibility / confidence predictions: NN sampling (visconf must stay in [0,1])
    x0, y0 = xy0[0, :, 0], xy0[0, :, 1]
    x0 = torch.clamp(x0, 0, W - 1).round().long()
    y0 = torch.clamp(y0, 0, H - 1).round().long()
    vis_predictions, confidence_predictions = [], []
    for vcl in all_visconf_preds:
        vps, cps = [], []
        for vc in vcl:
            vc = vc[:, :, :, y0, x0]   # B,S,2,N
            vps.append(vc[:, :, 0])
            cps.append(vc[:, :, 1])
        vis_predictions.append(vps)
        confidence_predictions.append(cps)

    # Slice GT into the same overlapping windows the model produced
    vis_gts, invis_gts, traj_gts, valids_gts = [], [], [], []
    for ind in range(0, T - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        valids_gts.append(valids[:, ind : ind + S])

    loss_context = (
        torch.amp.autocast("cuda", enabled=False)
        if cfg.loss_in_fp32
        else contextlib.nullcontext()
    )
    with loss_context:
        if cfg.loss_in_fp32:
            coord_predictions_f = [[c.float() for c in cp] for cp in coord_predictions]
            traj_gts_f = [t.float() for t in traj_gts]
            valids_gts_f = [v.float() for v in valids_gts]
            vis_gts_f = [v.float() for v in vis_gts]
            invis_gts_f = [v.float() for v in invis_gts]
            vis_predictions_f = [[v.float() for v in vp] for vp in vis_predictions]
            confidence_predictions_f = [[c.float() for c in cp] for cp in confidence_predictions]
        else:
            coord_predictions_f = coord_predictions
            traj_gts_f = traj_gts
            valids_gts_f = valids_gts
            vis_gts_f = vis_gts
            invis_gts_f = invis_gts
            vis_predictions_f = vis_predictions
            confidence_predictions_f = confidence_predictions

        if cfg.clamp_loss_probs:
            vis_predictions_f = [
                [torch.nan_to_num(v).clamp(1e-6, 1 - 1e-6) for v in vp]
                for vp in vis_predictions_f
            ]
            confidence_predictions_f = [
                [torch.nan_to_num(c).clamp(1e-6, 1 - 1e-6) for c in cp]
                for cp in confidence_predictions_f
            ]

        seq_loss_visible = loss.sequence_loss(
            coord_predictions_f,
            traj_gts_f,
            valids_gts_f,
            vis=vis_gts_f,
            gamma=0.8,
            use_huber_loss=cfg.use_huber_loss,
            loss_only_for_visible=True,
        )
        # PyTorch forbids BCELoss under autocast because its backward can be
        # numerically unsafe in fp16/bf16. Keep the rest of the loss path on the
        # configured precision, but compute probability losses in fp32.
        bce_context = (
            torch.amp.autocast(device.type, enabled=False)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with bce_context:
            coord_predictions_bce = [[c.float() for c in cp] for cp in coord_predictions_f]
            traj_gts_bce = [t.float() for t in traj_gts_f]
            vis_gts_bce = [v.float() for v in vis_gts_f]
            vis_predictions_bce = [[v.float() for v in vp] for vp in vis_predictions_f]
            confidence_predictions_bce = [
                [c.float() for c in cp] for cp in confidence_predictions_f
            ]
            confidence_loss = loss.sequence_prob_loss(
                coord_predictions_bce,
                confidence_predictions_bce,
                traj_gts_bce,
                vis_gts_bce,
            )
            vis_loss = loss.sequence_BCE_loss(vis_predictions_bce, vis_gts_bce)
        seq_loss_invisible = loss.sequence_loss(
            coord_predictions_f,
            traj_gts_f,
            valids_gts_f,
            vis=invis_gts_f,
            gamma=0.8,
            use_huber_loss=cfg.use_huber_loss,
            loss_only_for_visible=True,
        )

        total_loss = (
            seq_loss_visible.mean() * cfg.visible_loss_weight
            + seq_loss_invisible.mean() * cfg.invisible_loss_weight
            + vis_loss.mean()
            + confidence_loss.mean()
        )

    if cfg.skip_nonfinite_outputs and (torch.isnan(total_loss) or torch.isinf(total_loss)):
        print(f"[forward_batch_sparse] NaN/Inf total_loss, skipping batch")
        return None, None

    metrics: Dict[str, Any] = {}
    dn = dname[0]
    metrics["dname"] = dn
    metrics["seq_loss_visible"] = seq_loss_visible.mean().item()
    metrics["seq_loss_invisible"] = seq_loss_invisible.mean().item()
    metrics["vis_loss"] = vis_loss.item()
    metrics["conf_loss"] = confidence_loss.mean().item()
    thrs = [1, 2, 4, 8, 16]
    sx_ = (W - 1) / 255.0
    sy_ = (H - 1) / 255.0
    sc_pt = torch.tensor([sx_, sy_], device=device).reshape(1, 1, 1, 2)
    d_sum = 0.0
    for thr in thrs:
        d_ = (
            (torch.norm(trajs_e / sc_pt - trajs_g / sc_pt, dim=-1) < thr)
            .float()
            .mean()
            .item()
        )
        d_sum += d_
        metrics["d_%d" % thr] = d_
    metrics["d_avg"] = d_sum / len(thrs)
    metrics["total_loss"] = total_loss.item()

    return total_loss, metrics
