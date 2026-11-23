#!/usr/bin/env python3
"""Get the videos of RoboTrack-Bench's 42 Ego-Exo4D clips from Ego-Exo4D.

1. Get access to Ego-Exo4D (https://ego-exo4d-data.org) and install its downloader (pip install ego4d).
2. Download the takes listed in robotrack/egoexo4d_take_uids.txt (about 18 GB of the videos are used):
       egoexo -o /path/to/egoexo4d --parts takes --uids $(cat robotrack/egoexo4d_take_uids.txt)
3. Cut the clips into your RoboTrack-Bench folder:
       python -m robotrack.egoexo4d --egoexo-root /path/to/egoexo4d --out data/RoboTrack-Bench
   Each clip is written to <out>/<clip>/video.mp4, next to its point_tracks.npz.

Frame ranges in egoexo4d_clips.csv are zero-based, start-inclusive, end-exclusive indices into the 30 fps take video.
The Aria video is kept at its native 1408x1408; the exo video is resized to the listed size (1920x1080) with area
interpolation. Requires opencv-python; uses ffmpeg (H.264) for writing when it is on PATH, OpenCV's mp4v otherwise.
"""
import argparse, csv, shutil, subprocess, sys
from pathlib import Path
import cv2

MANIFEST = Path(__file__).resolve().parent / "egoexo4d_clips.csv"


def clip_names():
    """Names of the benchmark clips that come from Ego-Exo4D."""
    with open(MANIFEST) as f:
        return {r["clip"] for r in csv.DictReader(f)}


class Writer:
    def __init__(self, path, w, h, fps):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                          "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes()) if self.proc else self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        else:
            self.cv.release()


def cut(src, dst, start, end, w, h, fps):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    for _ in range(start):  # decode sequentially: seeking by frame number is not exact in every OpenCV build
        if not cap.grab():
            raise RuntimeError(f"{src} ends before frame {start}")
    tmp = dst.with_name("video.partial.mp4")
    out = Writer(tmp, w, h, fps)
    for i in range(start, end):
        ok, frame = cap.read()
        if not ok:
            out.close(); tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{src} ends at frame {i}, before {end}")
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        out.write(frame)
    out.close(); cap.release(); tmp.replace(dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egoexo-root", required=True, type=Path, help="folder that contains takes/<take_name>/frame_aligned_videos/")
    p.add_argument("--out", required=True, type=Path, help="RoboTrack-Bench folder; clips go to <out>/<clip>/video.mp4")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--clips", nargs="*", help="only these clips (default: all)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    rows = [r for r in csv.DictReader(open(args.manifest)) if not args.clips or r["clip"] in args.clips]
    missing = sorted({r["video_file"] for r in rows if not (args.egoexo_root / r["video_file"]).is_file()})
    if missing:
        sys.exit("missing take videos under --egoexo-root (download them first, see the top of this file):\n  " + "\n  ".join(missing))
    for k, r in enumerate(rows, 1):
        dst = args.out / r["clip"] / "video.mp4"
        if dst.exists() and not args.overwrite:
            print(f"[{k}/{len(rows)}] {r['clip']}: exists, skipped"); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cut(args.egoexo_root / r["video_file"], dst, int(r["start_frame"]), int(r["end_frame_exclusive"]), int(r["width"]), int(r["height"]), int(r["fps"]))
        n = int(cv2.VideoCapture(str(dst)).get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{k}/{len(rows)}] {r['clip']}: {n} frames, {r['width']}x{r['height']}" + ("" if n == int(r["num_frames"]) else f"  WARNING: expected {r['num_frames']}"))


if __name__ == "__main__":
    main()
