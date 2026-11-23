"""Evaluate AllTracker or RoboTracker on RoboTrack-Bench.

    python -m robotrack.eval_alltracker --checkpoint robotracker --data-root data/RoboTrack-Bench --out results/robotracker.json

--checkpoint takes a preset name (robotracker, alltracker), a URL or a local .pth file holding {"model": state_dict}.

Protocol (the one behind the numbers in the paper): frames resized to 384x512; every point is queried at its own first
visible frame and tracked to the end of the clip; points that share a query frame are tracked together in one dense pass
starting at that frame (the sliding-window pass for clips longer than 128 frames); 4 refinement iterations; a point is
predicted visible when visibility x confidence >= 0.6; fp32 with TF32 matmuls. Scores are TAP-Vid metrics averaged over
clips.
"""
import argparse

import torch

from robotrack import evaluate
from robotrack.hub import local_path
from robotrack.alltracker import Net

CHECKPOINTS = {
    "robotracker": "https://huggingface.co/allenai/RoboTracker/resolve/main/robotracker.pth",
    "alltracker": "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth",
}
IMAGE_SIZE = (384, 512)
ITERS = 4
SLIDING_THRESHOLD = 128


def load_model(checkpoint, device="cuda"):
    state = torch.load(local_path(CHECKPOINTS.get(checkpoint, checkpoint)), map_location="cpu", weights_only=True)
    model = Net(seqlen=16, init_weights=False)     # the checkpoint replaces every weight
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def track(model, frames, query_xy, query_frames):
    """Track query points through a video.

    frames        [T, H, W, 3] uint8
    query_xy      [N, 2] x, y at each point's query frame
    query_frames  [N] int
    Returns tracks [T, N, 2] and visibility probabilities [T, N] (zero before the query frame).
    """
    device = next(model.parameters()).device
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()[None]   # 1, T, 3, H, W
    T, H, W = video.shape[1], video.shape[3], video.shape[4]
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid = torch.stack([xs, ys]).float()[None, None]                                 # 1, 1, 2, H, W
    query_frames = torch.as_tensor(query_frames, device=device)
    query_xy = torch.as_tensor(query_xy, device=device)

    N = len(query_frames)
    tracks = torch.zeros(T, N, 2, device=device)
    vis_prob = torch.zeros(T, N, device=device)
    forward = model.forward_sliding if T > SLIDING_THRESHOLD else model
    for t in torch.unique(query_frames).tolist():
        idx = torch.nonzero(query_frames == t)[:, 0]
        # A dense pass from frame t gives every pixel's track; read off the pixels the points start at.
        traj_map = grid.repeat(1, T, 1, 1, 1)
        conf_map = torch.zeros_like(traj_map)
        if t < T - 1:
            flow, visconf, *_ = forward(video[:, t:], iters=ITERS, sw=None, is_training=False)
            traj_map[:, t:] = flow.to(device) + grid      # the sliding pass returns CPU tensors
            conf_map[:, t:] = visconf.to(device)
        x = query_xy[idx, 0].round().long().clamp(0, W - 1)
        y = query_xy[idx, 1].round().long().clamp(0, H - 1)
        tracks[:, idx] = traj_map[0, :, :, y, x].permute(0, 2, 1)
        conf = conf_map[0, :, :, y, x]                                               # T, 2, n
        vis_prob[:, idx] = conf[:, 0] * conf[:, 1]
    return tracks.cpu().numpy(), vis_prob.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="robotracker", help="preset name, URL or local .pth")
    evaluate.add_arguments(p)
    args = p.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.checkpoint)
    evaluate.run(args, lambda *a: track(model, *a), IMAGE_SIZE, args.checkpoint)


if __name__ == "__main__":
    main()
