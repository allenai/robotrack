"""RoboTrack-Bench: 307 real robot and human manipulation clips with 2,522 hand-annotated point tracks.

Each clip is a folder with
    video.mp4          the RGB video
    point_tracks.npz   trajs_2d    float32 [T, N, 2]  x, y in video pixels
                       visibility  float32 [T, N]     1 = visible, 0 = occluded
                       query_frames int32  [N]        first visible frame of each point
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robotrack import egoexo4d

HF_DATASET = "allenai/RoboTrack-Bench"


@dataclass
class Clip:
    name: str
    frames: np.ndarray      # [T, H, W, 3] uint8 RGB
    tracks: np.ndarray      # [T, N, 2] float32, x, y in pixels of `frames`
    visible: np.ndarray     # [T, N] bool
    source_size: tuple      # (H, W) of the original video


def download(local_dir):
    """Download the benchmark from the Hugging Face Hub and return its folder."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HF_DATASET, repo_type="dataset", local_dir=local_dir,
                                  allow_patterns=["*/video.mp4", "*/point_tracks.npz", "egoexo4d_clips.csv"]))


def list_clips(root, skip_missing_egoexo4d=False):
    """Clip folders under `root`, sorted by name.

    Get the Ego-Exo4D clips' videos with `python -m robotrack.egoexo4d` (see the README). A missing Ego-Exo4D video is an error unless `skip_missing_egoexo4d`; any other
    missing video is always an error.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark folder not found: {root}")
    clips = [d for d in sorted(root.iterdir()) if (d / "point_tracks.npz").is_file()]
    missing = [d.name for d in clips if not (d / "video.mp4").is_file()]
    egoexo = egoexo4d.clip_names()
    other = [m for m in missing if m not in egoexo]
    if other:
        raise FileNotFoundError(f"{len(other)} clips have labels but no video.mp4 (incomplete download?): {', '.join(other[:5])}")
    if missing and not skip_missing_egoexo4d:
        raise FileNotFoundError(
            f"{len(missing)} Ego-Exo4D clips have no video. Get them from Ego-Exo4D with\n"
            f"    python -m robotrack.egoexo4d --egoexo-root /path/to/egoexo4d --out {root}\n"
            f"(see robotrack/egoexo4d.py), or pass --skip-missing-egoexo4d to evaluate without them "
            f"(scores are then not comparable with the paper).")
    return [d for d in clips if d.name not in missing]


def read_frames(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_clip(clip_dir, size=None):
    """Load one clip. With `size=(H, W)`, frames are resized and the tracks scaled to match.

    Points visible in fewer than two frames are dropped (nothing to evaluate after the query frame).
    """
    clip_dir = Path(clip_dir)
    labels = np.load(clip_dir / "point_tracks.npz")
    tracks = labels["trajs_2d"].astype(np.float32)
    visible = labels["visibility"] > 0.5
    frames = read_frames(clip_dir / "video.mp4", max_frames=tracks.shape[0])
    if not frames:
        raise RuntimeError(f"could not read {clip_dir / 'video.mp4'}")

    T = min(len(frames), tracks.shape[0])
    frames, tracks, visible = frames[:T], tracks[:T], visible[:T]
    keep = np.isfinite(tracks).all(axis=(0, 2)) & (visible.sum(0) > 1)
    tracks, visible = tracks[:, keep], visible[:, keep]

    H0, W0 = frames[0].shape[:2]
    if size is not None:
        H, W = size
        frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames]
        tracks = tracks * np.array([(W - 1) / max(W0 - 1, 1), (H - 1) / max(H0 - 1, 1)], np.float32)
    return Clip(clip_dir.name, np.stack(frames), tracks, visible, (H0, W0))
