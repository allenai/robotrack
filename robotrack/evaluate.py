"""Shared RoboTrack-Bench evaluation loop: the tracker-specific scripts (eval_alltracker, eval_cowtracker) supply a
`track(frames, query_xy, query_frames)` function and the image size they run at; this module loads the clips, queries
every point at its first visible frame, scores the predictions and writes the results."""
import json
import time
from pathlib import Path

import numpy as np

from robotrack import bench
from robotrack.metrics import tapvid_metrics

VIS_THRESHOLD = 0.6   # visible when visibility x confidence >= 0.6 (AllTracker's and CoTracker3's rule)


def add_arguments(p):
    p.add_argument("--data-root", type=Path, required=True, help="RoboTrack-Bench folder (see --download)")
    p.add_argument("--download", action="store_true", help="download the benchmark into --data-root first")
    p.add_argument("--out", type=Path, required=True, help="results JSON (per-clip and mean scores)")
    p.add_argument("--save-predictions", type=Path, help="also write one .npz of predicted tracks per clip here")
    p.add_argument("--clips", nargs="*", help="evaluate only these clip names")
    p.add_argument("--skip-missing-egoexo4d", action="store_true",
                   help="evaluate without the Ego-Exo4D clips you have not rebuilt (scores not comparable with the paper)")


def evaluate_clip(track, clip, image_size):
    query_frames = clip.visible.argmax(0)
    query_xy = clip.tracks[query_frames, np.arange(clip.tracks.shape[1])]
    tracks, vis_prob = track(clip.frames, query_xy, query_frames)
    scores = tapvid_metrics(query_frames, ~clip.visible.T, clip.tracks.transpose(1, 0, 2),
                            (vis_prob < VIS_THRESHOLD).T, tracks.transpose(1, 0, 2), image_size)
    return scores, tracks, vis_prob


def run(args, track, image_size, checkpoint_name):
    """Evaluate `track` on every clip of the benchmark and write args.out."""
    if args.download:
        bench.download(args.data_root)
    clips = bench.list_clips(args.data_root, skip_missing_egoexo4d=args.skip_missing_egoexo4d)
    labelled = sorted(d.name for d in args.data_root.iterdir() if (d / "point_tracks.npz").is_file())
    skipped = sorted(set(labelled) - {c.name for c in clips})
    if skipped:
        print(f"WARNING: skipping {len(skipped)} Ego-Exo4D clips without video; scores are not comparable with the paper")
    if args.clips:
        clips = [c for c in clips if c.name in set(args.clips)]
    if not clips:
        raise SystemExit(f"no clips found in {args.data_root}")
    if args.save_predictions:
        args.save_predictions.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for i, clip_dir in enumerate(clips):
        clip = bench.load_clip(clip_dir, image_size)
        scores, tracks, vis_prob = evaluate_clip(track, clip, image_size)
        rows.append({"clip": clip.name, "n_points": int(clip.tracks.shape[1]), **scores})
        if args.save_predictions:
            H0, W0 = clip.source_size
            to_source = np.array([(W0 - 1) / (image_size[1] - 1), (H0 - 1) / (image_size[0] - 1)], np.float32)
            np.savez_compressed(args.save_predictions / f"{clip.name}.npz", tracks=tracks * to_source,
                                visibility=vis_prob >= VIS_THRESHOLD, visibility_prob=vis_prob)
        print(f"[{i + 1}/{len(clips)}] {clip.name}: AJ {scores['aj'] * 100:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    mean = {k: 100 * float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("clip", "n_points")}
    n_points = sum(r["n_points"] for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoint": checkpoint_name, "n_clips": len(rows), "n_points": n_points,
                                    "skipped_clips": skipped, "mean": mean, "clips": rows}, indent=1))
    print(f"RoboTrack-Bench ({len(rows)} clips, {n_points} points): "
          f"AJ {mean['aj']:.2f}  <δavg {mean['d_avg']:.2f}  OA {mean['oa']:.2f}  -> {args.out}")
