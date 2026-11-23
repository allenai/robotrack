"""Build the training manifests for generated point-tracking data.

    python datagen/build_manifests.py --root out/robotrack15m

Reads every finished house (<root>/houses/house_*/generation_result.json, as written by generate_robotrack15m.py; or
<root>/*/generation_result.json for houses generated one by one with generate_house.py) and writes
<root>/manifests/metadata_long.jsonl and metadata_short.jsonl, which robotrack.training reads (set
robotrack15m_root=<root>). Each line names one camera video and its tracks, relative to <root>.

The split follows RoboTrack-15M's: the unit is one episode (its cameras stay together); a shuffled half of the episodes
with at least --long-frames frames goes to the long manifest (56-frame training clips), everything else to the short
one (24-frame clips). Episodes shorter than --min-frames are left out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def find_houses(root: Path) -> list[Path]:
    houses = sorted(root.glob("houses/*/generation_result.json")) or sorted(root.glob("*/generation_result.json"))
    if not houses:
        raise SystemExit(f"no generation_result.json under {root}/houses/ or {root}/")
    return [p.parent for p in houses]


def episode_rows(root: Path, house: Path, result: dict, episode: dict) -> list[dict]:
    rows = []
    for camera, rec in sorted(episode["cameras"].items()):
        video, tracks = house / rec["rgb_video"], house / rec["point_tracks"]
        for path in (video, tracks):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append(dict(file_name=str(video.relative_to(root)), point_tracks_file=str(tracks.relative_to(root)),
                         num_frames=int(rec.get("frames", rec.get("num_frames"))), house_id=int(result["house_id"]),
                         episode=int(episode["episode"]), camera=camera))
    if len({r["num_frames"] for r in rows}) != 1:
        raise ValueError(f"camera videos of {house} episode {episode['episode']} differ in length")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="generation output (holds houses/)")
    p.add_argument("--out", type=Path, help="manifest folder (default <root>/manifests)")
    p.add_argument("--seed", type=int, default=42, help="the split shuffle (RoboTrack-15M used 42)")
    p.add_argument("--long-frames", type=int, default=56, help="episodes this long can go to the long manifest")
    p.add_argument("--min-frames", type=int, default=24, help="shorter episodes are left out")
    a = p.parse_args()
    root, out = a.root.resolve(), (a.out or a.root / "manifests")

    units, skipped = {}, 0
    for house in find_houses(root):
        result = json.loads((house / "generation_result.json").read_text())
        for episode in result.get("episodes") or [dict(episode=0, cameras=result["cameras"])]:
            rows = episode_rows(root, house, result, episode)
            if rows[0]["num_frames"] < a.min_frames:
                skipped += 1
                continue
            units[f"{house.name}/episode_{episode['episode']:08d}"] = rows
    eligible = [u for u, rows in units.items() if rows[0]["num_frames"] >= a.long_frames]
    random.Random(a.seed).shuffle(eligible)
    long_units = set(eligible[:len(eligible) // 2])
    splits = {"long": [], "short": []}
    for unit, rows in units.items():
        splits["long" if unit in long_units else "short"].extend(rows)
    if not all(splits.values()):
        raise SystemExit(f"need episodes in both manifests (long: >= {a.long_frames} frames); got "
                         f"{len(splits['long'])} long and {len(splits['short'])} short videos")

    out.mkdir(parents=True, exist_ok=True)
    summary = dict(root=str(root), seed=a.seed, long_frames=a.long_frames, min_frames=a.min_frames,
                   skipped_short_episodes=skipped, splits={})
    for name, rows in splits.items():
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (out / f"metadata_{name}.jsonl").write_text(text)
        summary["splits"][name] = dict(videos=len(rows), houses=len({r["house_id"] for r in rows}),
                                       min_frames=min(r["num_frames"] for r in rows),
                                       max_frames=max(r["num_frames"] for r in rows),
                                       sha256=hashlib.sha256(text.encode()).hexdigest())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{out}: long {len(splits['long'])} videos, short {len(splits['short'])} videos"
          + (f" ({skipped} episodes under {a.min_frames} frames left out)" if skipped else ""))


if __name__ == "__main__":
    main()
