"""Evaluate RoboTracker-Warp or CoWTracker on RoboTrack-Bench.

    git clone --recursive https://github.com/facebookresearch/cowtracker /path/to/cowtracker
    python -m robotrack.eval_cowtracker --cowtracker-repo /path/to/cowtracker --checkpoint robotracker-warp \
        --data-root data/RoboTrack-Bench --out results/robotracker_warp.json

The CoWTracker model code is not part of this repository: it is under the FAIR Noncommercial Research License, so
you install it from Meta's repository (with its VGGT and Depth-Anything-V2 submodules, and xformers) and point
--cowtracker-repo at it. RoboTracker-Warp's weights are CC-BY-NC 4.0, like CoWTracker's.

--checkpoint takes robotracker-warp, cowtracker (the released CoWTracker), a URL or a local .pth holding
{"model": state_dict}.

Protocol (the one behind the numbers in the paper; CoWTracker releases no evaluation code, so it follows the paper
and its demo): frames resized to 336x560; fp16 weights under fp16 autocast, pixel values 0..255, 5 warping
iterations; every point is queried at its own first visible frame; points that share a query frame are tracked
together in one dense pass from that frame through CoWTrackerWindowed (100-frame windows, stride 100, up to 10 memory
frames); a point is predicted visible when visibility x confidence >= 0.6.
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import torch

from robotrack import evaluate
from robotrack.hub import local_path

CHECKPOINTS = {
    "robotracker-warp": "https://huggingface.co/allenai/RoboTracker-Warp/resolve/main/robotracker_warp.pth",
    "cowtracker": None,   # CoWTracker's own loader downloads the released weights (facebook/cowtracker)
}
IMAGE_SIZE = (336, 560)


def load_model(cowtracker_repo, checkpoint, device="cuda"):
    sys.path.insert(0, str(Path(cowtracker_repo).resolve()))
    try:
        from cowtracker import CoWTrackerWindowed
    except ImportError as e:
        raise SystemExit(f"cannot import CoWTracker from {cowtracker_repo} ({e}); clone "
                         "https://github.com/facebookresearch/cowtracker with --recursive and install its requirements")
    path = CHECKPOINTS.get(checkpoint, checkpoint)
    if path is not None:
        path = local_path(path)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        model = CoWTrackerWindowed.from_checkpoint(path, device=device, dtype=torch.float16)
    # from_checkpoint loads with strict=False and only prints the result; a key mismatch would leave random weights.
    message = [line for line in log.getvalue().splitlines() if line.startswith("Load message")]
    if not message or "All keys matched successfully" not in message[0]:
        raise RuntimeError(f"checkpoint keys do not match the model: {message}")
    return model


@torch.no_grad()
def track(model, frames, query_xy, query_frames):
    """Track query points through a video (same interface as eval_alltracker.track)."""
    device = next(model.parameters()).device
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()          # T, 3, H, W in 0..255
    T, H, W = video.shape[0], video.shape[2], video.shape[3]
    query_frames = torch.as_tensor(query_frames, device=device)
    query_xy = torch.as_tensor(query_xy, device=device)

    N = len(query_frames)
    tracks = torch.zeros(T, N, 2, device=device)
    vis_prob = torch.zeros(T, N, device=device)
    for t in torch.unique(query_frames).tolist():
        idx = torch.nonzero(query_frames == t)[:, 0]
        x = query_xy[idx, 0].round().long().clamp(0, W - 1)
        y = query_xy[idx, 1].round().long().clamp(0, H - 1)
        tracks[:, idx] = torch.stack([x, y], -1).float()   # where the model does not run: the pixel, not visible
        if t < T - 1:
            # A dense pass from frame t gives every pixel's track; read off the pixels the points start at.
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16), contextlib.redirect_stdout(io.StringIO()):
                pred = model(video[t:])                                            # prints every window otherwise
            track_map, vis_map, conf_map = pred["track"][0], pred["vis"][0], pred["conf"][0]   # S, H, W(, 2)
            tracks[t:, idx] = track_map[:, y, x].float()
            vis_prob[t:, idx] = vis_map[:, y, x].float() * conf_map[:, y, x].float()
            del pred, track_map, vis_map, conf_map
    return tracks.cpu().numpy(), vis_prob.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cowtracker-repo", default=os.environ.get("COWTRACKER_REPO"), required="COWTRACKER_REPO" not in os.environ,
                   help="clone of facebookresearch/cowtracker (or set COWTRACKER_REPO)")
    p.add_argument("--checkpoint", default="robotracker-warp", help="preset name, URL or local .pth")
    evaluate.add_arguments(p)
    args = p.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(args.cowtracker_repo, args.checkpoint)
    evaluate.run(args, lambda *a: track(model, *a), IMAGE_SIZE, args.checkpoint)


if __name__ == "__main__":
    main()
