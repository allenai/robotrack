"""RoboTrack-15M generation as an add-on to the official MolmoSpaces (https://github.com/allenai/molmospaces).

Nothing in MolmoSpaces is modified. This module plugs into its data-generation pipeline from outside:
  - RoboTrack15MPickConfig: MolmoSpaces' Franka pick task with DROID-style cameras, plus the point-track settings.
  - RoboTrackRunner: a ParallelRolloutRunner whose run_single_rollout wraps the task's reset() and step_chunk() to record
    every video frame (body poses, and per camera the depth render and camera matrices; every 20th frame, candidate
    points sampled from the segmentation). After the rollout it selects the tracks and replays the recorded frames to
    get their 2D positions and visibility, and attaches them to the episode history.
  - save_house_trajectories: replaces MolmoSpaces' saver for this config only; writes each episode's camera videos and
    one point-track .npz per camera, and no HDF5 trajectory file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, FrankaDroidCameraSystem
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation import pipeline
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.save_utils import save_videos_from_raw_observations

import point_tracking as pt

log = logging.getLogger(__name__)
HISTORY_KEY = "robotrack_point_tracks"


@register_config("RoboTrack15MPick")
class RoboTrack15MPickConfig(PickBaseConfig):
    """Franka pick with a wrist and a base-mounted exo camera, and point tracks on the robot and movable objects."""

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: CameraSystemConfig | None = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "robotrack15m"
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler, house_inds=[0], samples_per_house=1)
    num_workers: int = 1
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = True
    point_track_num_points: int = 5000                # per camera
    point_track_query_interval: int = 20              # sample candidate points every this many frames
    point_track_exclude_robot_links: list[str] = ["base", "fr3_link0"]   # the static base: no tracks
    point_track_occlusion_tolerance: float = 0.03     # metres behind the rendered depth that still counts as visible

    @property
    def tag(self) -> str:
        return "robotrack15m_pick"


class _Recorder:
    """Records one episode and turns it into point tracks."""

    def __init__(self, task, seed: int, cfg: RoboTrack15MPickConfig):
        self.task, self.seed, self.cfg = task, seed, cfg
        model = task.env.mj_model
        self.cameras = [c.name for c in cfg.camera_config.cameras]
        self.w, self.h = cfg.camera_config.img_resolution
        self.bodies = pt.trackable_body_ids(model)
        excluded = set()
        for link in cfg.point_track_exclude_robot_links:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_0/{link}")
            if bid < 0 or bid not in self.bodies:
                raise ValueError(f"cannot exclude robot_0/{link}: not a trackable body")
            excluded.add(int(bid))
        self.excluded = np.array(sorted(excluded), np.int64)
        self.robot = np.array(sorted(pt.robot_body_ids(model) - excluded), np.int64)
        self.xpos, self.xmat = [], []
        self.depth = {c: [] for c in self.cameras}
        self.mats = {c: [] for c in self.cameras}
        self.cand = {c: ([], []) for c in self.cameras}

    def record(self):
        """One video frame: called after reset() and after every step_chunk()."""
        env = self.task.env
        frame = len(self.xpos)
        self.xpos.append(env.current_data.xpos.copy())
        self.xmat.append(env.current_data.xmat.copy())
        for name in self.cameras:
            camera = env.camera_manager.registry[name]
            depth = env.render_depth_frame(name)
            self.depth[name].append(depth.copy())
            self.mats[name].append(pt.camera_matrices(camera, self.w, self.h))
            if frame % self.cfg.point_track_query_interval == 0:
                seg = env.render_segmentation_frame(name)
                lc, bi, _ = pt.sample_from_image(env.mj_model, env.current_data, camera, self.w, self.h, depth, seg,
                                                 self.cfg.point_track_num_points, self.seed + frame, self.bodies)
                if len(lc):
                    self.cand[name][0].append(lc)
                    self.cand[name][1].append(bi)

    def finish(self) -> dict:
        """Select up to point_track_num_points candidates per camera and track them through every recorded frame."""
        n = self.cfg.point_track_num_points
        rng = np.random.RandomState(self.seed + 9999)           # one stream, cameras in order
        out = {}
        for ci, name in enumerate(self.cameras):
            if not self.cand[name][0]:
                continue
            lc, bi = np.concatenate(self.cand[name][0]), np.concatenate(self.cand[name][1])
            sel = rng.choice(len(lc), min(n, len(lc)), replace=False)
            if len(self.excluded):
                # Picks on the excluded base links move to other robot bodies, from a separate stream.
                on_excluded = np.isin(bi[sel], self.excluded)
                pool = np.setdiff1d(np.flatnonzero(np.isin(bi, self.robot)), sel)
                move = np.random.RandomState(self.seed + 424242 + ci)
                repl = move.choice(pool, min(int(on_excluded.sum()), len(pool)), replace=False)
                slots = np.flatnonzero(on_excluded)
                sel[slots[:len(repl)]] = repl
                sel = np.delete(sel, slots[len(repl):])
            lc, bi = lc[sel], bi[sel]
            frames = [pt.track_frame(self.xpos[f], self.xmat[f], lc, bi, *self.mats[name][f], self.w, self.h,
                                     self.depth[name][f], self.cfg.point_track_occlusion_tolerance)
                      for f in range(len(self.xpos))]
            vis = np.stack([f[1] for f in frames])
            out[name] = dict(trajs_2d=np.stack([f[0] for f in frames]), visibility=vis,
                             points_3d=np.stack([f[2] for f in frames]), body_ids=bi,
                             intrinsics=self.mats[name][0][1], query_frames=np.argmax(vis > 0.5, axis=0))
        return out


class RoboTrackRunner(pipeline.ParallelRolloutRunner):
    @staticmethod
    def run_single_rollout(episode_seed, task, policy, *args, **kwargs):
        cfg = task.config
        if not isinstance(cfg, RoboTrack15MPickConfig):
            return pipeline.ParallelRolloutRunner.run_single_rollout(episode_seed, task, policy, *args, **kwargs)
        state = {}
        orig_reset, orig_step, orig_history = task.reset, task.step_chunk, task.get_history

        def reset(*a, **k):
            result = orig_reset(*a, **k)
            state["rec"] = _Recorder(task, episode_seed, cfg)
            state["rec"].record()
            return result

        def step_chunk(*a, **k):
            result = orig_step(*a, **k)
            state["rec"].record()
            return result

        task.reset, task.step_chunk = reset, step_chunk          # instance attributes shadow the methods
        try:
            success = pipeline.ParallelRolloutRunner.run_single_rollout(episode_seed, task, policy, *args, **kwargs)
        finally:
            del task.reset, task.step_chunk
        tracks = state["rec"].finish() if "rec" in state else {}

        def get_history():
            history = orig_history()
            history[HISTORY_KEY] = tracks
            return history

        task.get_history = get_history
        return success


_molmospaces_save = pipeline.save_house_trajectories


def save_house_trajectories(worker_logger, house_raw_histories, house_output_dir, exp_config, batch_suffix,
                            datagen_profiler=None, batch_num=None, total_batches=None):
    if not isinstance(exp_config, RoboTrack15MPickConfig):
        return _molmospaces_save(worker_logger, house_raw_histories, house_output_dir, exp_config, batch_suffix,
                                 datagen_profiler, batch_num, total_batches)
    for idx, episode in enumerate(house_raw_histories):
        history = episode.pop("history")
        tracks = history.pop(HISTORY_KEY, None) or {}
        observations = history.get("observations", [])
        if observations:
            save_videos_from_raw_observations([step[0] for step in observations], house_output_dir, exp_config.fps,
                                              episode_idx=idx, save_file_suffix=batch_suffix,
                                              sensor_suite=episode.get("sensor_suite"))
        for camera, data in tracks.items():
            pt.save_point_tracks(house_output_dir / f"episode_{idx:08d}_{camera}_point_tracks.npz", **data)
        worker_logger.info(f"saved episode {idx}: videos and point tracks for {sorted(tracks)}")


pipeline.save_house_trajectories = save_house_trajectories   # process_single_house looks it up in the module
