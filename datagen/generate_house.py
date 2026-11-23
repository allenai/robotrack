"""Generate the RoboTrack-15M episodes of one ProcTHOR house.

Needs the official MolmoSpaces (https://github.com/allenai/molmospaces; the third_party/molmospaces submodule),
unmodified: datagen/molmospaces_ext.py adds the point tracking from outside. Also a GPU with EGL, and ffprobe. Usually run through generate_robotrack15m.py; standalone:

    MLSPACES_LICENSE_POLICY=commercial_safe MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        python datagen/generate_house.py --output-dir out/house_1386 --house-id 1386 --seed 158865086

The recipe (the defaults below are RoboTrack-15M's): a Franka arm picks one THOR object per episode (5 episodes per
house), filmed by a wrist camera (56.74 deg) and a base-mounted third-person camera (71 deg, the DROID setup) at
624x352. Each camera gets up to 5,000 point tracks drawn from candidates every 20 frames on the robot and the movable
objects (none on walls, floors or fixed furniture, none on the static robot base), each queried at its first visible
frame; a point is visible when its depth matches the rendered depth at its nearest pixel to within 3 cm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

CAMERAS = ("wrist_camera", "exo_camera_1")
# Asset versions the dataset was generated with; MolmoSpaces downloads them on first use.
ASSET_VERSIONS = {
    "robots": {"franka_droid": "20260127"},
    "scenes": {"procthor-10k-train": "20251122_with_occupancy", "refs": "20250923"},
    "objects": {"thor": "20251117", "objathor_metadata": "20260129"},
    "grasps": {"droid": "20251116"}, "test_data": {}, "benchmarks": {},
}


def build_config(args):
    from molmo_spaces import molmo_spaces_constants
    from molmo_spaces.configs.camera_configs import FrankaDroidCameraSystem
    from molmospaces_ext import RoboTrack15MPickConfig

    molmo_spaces_constants.DATA_TYPE_TO_SOURCE_TO_VERSION.clear()
    molmo_spaces_constants.DATA_TYPE_TO_SOURCE_TO_VERSION.update(copy.deepcopy(ASSET_VERSIONS))
    config = RoboTrack15MPickConfig()
    cameras = FrankaDroidCameraSystem()            # wrist + base-mounted exo, fixed, no noise
    cameras.img_resolution = tuple(args.resolution)
    cameras.cameras = copy.deepcopy(cameras.cameras[:2])
    assert [c.name for c in cameras.cameras] == list(CAMERAS)
    config.camera_config = cameras
    config.output_dir = args.output_dir / "generated"
    config.seed = args.seed
    config.num_workers = 1
    config.datagen_profiler = False
    config.use_wandb = False
    config.filter_for_successful_trajectories = True
    config.scene_dataset = "procthor-10k"
    config.data_split = "train"
    tsc = config.task_sampler_config
    tsc.dataset_name = "procthor-10k"
    tsc.added_pickup_objects = None
    tsc.added_pickup_class_rank = None
    tsc.num_added_pickups = 0
    tsc.objaverse_oversampling_factor = 1
    tsc.house_inds = [args.house_id]
    tsc.samples_per_house = args.episodes
    tsc.max_total_attempts_multiplier = args.max_attempts
    # Point tracks: candidates every 20 frames over the robot and free objects, picked uniformly from the pool;
    # picks on the static base (the wooden platform and fr3_link0) move to other robot bodies.
    config.point_track_num_points = args.num_points
    config.point_track_query_interval = args.query_interval
    config.point_track_occlusion_tolerance = args.occlusion_tolerance
    config.point_track_exclude_robot_links = list(args.exclude_robot_links)
    return config


def frame_count(video):
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_read_frames:frame=best_effort_timestamp_time", "-of", "json", str(video)],
        text=True, timeout=120))
    stream = probe["streams"][0]
    return (int(stream["nb_read_frames"]), stream["width"], stream["height"]), probe["frames"]


def check_tracks(path, num_points, resolution):
    """The checks every released file passed. Returns a per-camera summary."""
    width, height = resolution
    with np.load(path) as tracks:
        xy, vis, qf = tracks["trajs_2d"], tracks["visibility"], tracks["query_frames"]
        t, n, _ = xy.shape
        assert 0 < n <= num_points and t > 1
        assert vis.shape == (t, n) and tracks["points_3d"].shape == (t, n, 3) and tracks["body_ids"].shape == (n,)
        seen = (vis > .5).any(axis=0)
        # A few tracks (about 2 to 11 per 5,000) are never visible under the depth test; their query frame is 0 and
        # training never uses them.
        assert (~seen).mean() < 0.01, f"{int((~seen).sum())} never-visible tracks in {path}"
        np.testing.assert_array_equal(qf, np.argmax(vis > .5, axis=0))
        assert (vis[qf[seen], np.arange(n)[seen]] == 1).all()
        # The depth test rounds to the nearest pixel, so a visible point can sit up to 1 px outside the frame.
        near = (xy[..., 0] > -1) & (xy[..., 0] < width + 1) & (xy[..., 1] > -1) & (xy[..., 1] < height + 1)
        assert np.isfinite(xy[vis > .5]).all() and near[vis > .5].all()
        return dict(frames=t, points=n, never_visible_tracks=int((~seen).sum()), mean_visibility=float(vis.mean()))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--house-id", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--num-points", type=int, default=5000, help="tracks per camera")
    p.add_argument("--max-attempts", type=int, default=6, help="rollout attempts per wanted episode")
    p.add_argument("--exclude-robot-links", nargs="*", default=["base", "fr3_link0"],
                   help="robot links that get no tracks (default: the static base)")
    p.add_argument("--resolution", nargs=2, type=int, default=[624, 352], metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--query-interval", type=int, default=20, help="sample candidate points every this many frames")
    p.add_argument("--occlusion-tolerance", type=float, default=0.03,
                   help="metres behind the rendered depth that still count as visible")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe not found on PATH (it checks the videos)")
    args.output_dir.mkdir(parents=True)

    from molmospaces_ext import RoboTrackRunner
    config = build_config(args)
    success, total = RoboTrackRunner(config).run()
    if success != args.episodes:
        raise RuntimeError(f"expected {args.episodes} successful episodes, got {success} of {total}")

    house = config.output_dir / f"house_{args.house_id}"
    episodes = []
    for ep in range(args.episodes):
        cams, stamps = {}, {}
        for cam in CAMERAS:
            npz = house / f"episode_{ep:08d}_{cam}_point_tracks.npz"
            mp4 = house / f"episode_{ep:08d}_{cam}_batch_1_of_1.mp4"
            cams[cam] = dict(point_tracks=str(npz.relative_to(args.output_dir)),
                             rgb_video=str(mp4.relative_to(args.output_dir)), **check_tracks(npz, args.num_points, args.resolution))
            shape, stamps[cam] = frame_count(mp4)
            assert shape == (cams[cam]["frames"], *args.resolution), (mp4, shape)
        assert stamps[CAMERAS[0]] == stamps[CAMERAS[1]], "the two cameras' videos are not synchronized"
        episodes.append(dict(episode=ep, cameras=cams))
    if len(list(house.glob("*point_tracks.npz"))) != 2 * args.episodes or len(list(house.glob("*.mp4"))) != 2 * args.episodes:
        raise RuntimeError(f"unexpected files in {house}")

    summary = dict(house_id=args.house_id, seed=args.seed, episodes_per_house=args.episodes, successful=success,
                   attempts=total, points_per_camera=args.num_points, excluded_robot_links=args.exclude_robot_links,
                   cameras=list(CAMERAS), resolution=list(args.resolution), scene_dataset="procthor-10k", split="train",
                   object_assets="thor", sampling=f"legacy image sampler, candidates every {args.query_interval} frames, no background",
                   visibility=f"nearest-pixel depth test, {args.occlusion_tolerance * 100:g} cm", asset_versions=ASSET_VERSIONS, episodes=episodes)
    (args.output_dir / "generation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = sorted(f for f in args.output_dir.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(f)}  {f.relative_to(args.output_dir)}\n" for f in files))
    print(f"house {args.house_id}: {success} episodes ({total} attempts)", flush=True)


if __name__ == "__main__":
    main()
