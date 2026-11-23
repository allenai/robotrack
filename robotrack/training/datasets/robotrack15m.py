"""RoboTrack-15M training loader.

RoboTrack-15M is manifest based: each line of manifests/metadata_{long,short}.jsonl names a video (`file_name`) and its
tracks (`point_tracks_file`, an .npz with `trajs_2d` [T, N, 2] and `visibility` [T, N]), relative to the dataset root.
Every draw takes a random window of `seq_len` frames, applies AllTracker's augmentations, and returns up to
`traj_per_sample * traj_max_factor` tracks that are visible in the window's first frame (the query frame).
"""
import json
import os
import random

import cv2
import numpy as np
import torch

from robotrack.alltracker import data
from robotrack.training.datasets.pointdataset import PointDataset


class RoboTrack15MDataset(PointDataset):
    def __init__(self, data_root, manifest, crop_size=(384, 512), seq_len=24, traj_per_sample=256, traj_max_factor=24,
                 use_augs=True, shuffle=True, split_seed=42, far_position_distance=1000.0, dataset_name="robotrack15m"):
        super().__init__(data_root=data_root, crop_size=crop_size, seq_len=seq_len, traj_per_sample=traj_per_sample,
                         use_augs=use_augs)
        self.dname = "%s%d" % (dataset_name, seq_len)
        self.traj_max_factor = traj_max_factor
        self.far_position_distance = float(far_position_distance)

        manifest_path = os.path.join(data_root, manifest)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"RoboTrack-15M manifest not found: {manifest_path}")
        samples = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    video, tracks = os.path.join(data_root, rec["file_name"]), os.path.join(data_root, rec["point_tracks_file"])
                    if os.path.isfile(video) and os.path.isfile(tracks):
                        samples.append((video, tracks))
        # Fixed shuffle, then the run's own shuffle (seeded per rank), as in the runs behind the paper.
        random.Random(split_seed).shuffle(samples)
        if shuffle:
            random.shuffle(samples)
        self.samples = samples
        print(f"found {len(samples)} RoboTrack-15M videos in {manifest_path}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _coordinate_validity(trajs):
        return np.isfinite(trajs).all(axis=-1) & (np.abs(trajs.astype(np.float64)).sum(axis=-1) < 100000)

    def _read_frames(self, path, max_frames):
        cap, frames = cv2.VideoCapture(path), []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def getitem_helper(self, index):
        video_path, track_path = self.samples[index]
        with np.load(track_path) as labels:
            trajs = labels["trajs_2d"].astype(np.float32)        # T, N, 2
            visibs = labels["visibility"].astype(np.float32)     # T, N
        rgbs = self._read_frames(video_path, max_frames=trajs.shape[0])
        if not rgbs:
            return self._fake_sample(), False
        T = min(len(rgbs), trajs.shape[0])
        rgbs, trajs, visibs = np.stack(rgbs[:T]), trajs[:T], visibs[:T]
        if T < self.seq_len:
            return self._fake_sample(), False
        H, W = rgbs.shape[1:3]

        # Off-screen positions stay valid targets but are not visible; invalid coordinates are zeroed and masked.
        valids = self._coordinate_validity(trajs)
        off = (trajs[..., 0] < 0) | (trajs[..., 0] > W - 1) | (trajs[..., 1] < 0) | (trajs[..., 1] > H - 1)
        visibs[off | ~valids] = 0
        trajs[~valids] = 0
        if trajs.shape[1] == 0:
            return self._fake_sample(), False

        # A random window of seq_len frames.
        seq_len = self.seq_len
        if seq_len < T:
            start = np.random.randint(0, T - seq_len + 1)
            rgbs, trajs, visibs, valids = (a[start:start + seq_len] for a in (rgbs, trajs, visibs, valids))
        rgbs, trajs, visibs, valids = rgbs[:seq_len], trajs[:seq_len], visibs[:seq_len], valids[:seq_len]

        # AllTracker's augmentations: time reversal, then photometric and spatial augs (or a plain crop).
        if self.use_augs and np.random.rand() < 0.5:
            rgbs, trajs, visibs, valids = (np.flip(a, axis=[0]).copy() for a in (rgbs, trajs, visibs, valids))
        if self.use_augs and np.random.rand() < 0.98:
            rgbs, trajs, visibs = self.add_photometric_augs(rgbs, trajs, visibs, replace=False)
            rgbs, trajs = self.add_spatial_augs(rgbs, trajs, visibs, self.crop_size)
        else:
            rgbs, trajs = self.crop(rgbs, trajs, self.crop_size)
        valids &= self._coordinate_validity(trajs)
        trajs[~valids] = 0
        visibs[~valids] = 0
        visibs[(trajs[..., 0] < 0) | (trajs[..., 0] > self.crop_size[1] - 1)
               | (trajs[..., 1] < 0) | (trajs[..., 1] > self.crop_size[0] - 1)] = 0

        # Queries are the tracks visible in the window's first frame.
        first = visibs[0] > 0.5
        trajs, visibs, valids = trajs[:, first], visibs[:, first], valids[:, first]
        trajs, visibs, valids = (torch.from_numpy(a.copy()).float() for a in (trajs, visibs, valids))

        # Positions far from the crop centre are not learnable targets: mask those frames, keep the track.
        centre = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        far = torch.linalg.vector_norm(trajs - centre, dim=-1) >= self.far_position_distance
        valids[far], visibs[far], trajs[far] = 0, 0, 0
        N = trajs.shape[1]
        if N < 1:
            return self._fake_sample(), False

        keep = torch.randperm(N)[: self.traj_per_sample * self.traj_max_factor]
        trajs, visibs, valids = trajs[:, keep], visibs[:, keep], valids[:, keep]
        gotit = trajs.shape[1] >= self.traj_per_sample
        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()
        return data.VideoData(video=rgbs, trajs=trajs, visibs=visibs, valids=valids, dname=self.dname), gotit

    def _fake_sample(self):
        S, N = self.seq_len, self.traj_per_sample
        return data.VideoData(video=torch.zeros((S, 3, *self.crop_size)), trajs=torch.zeros((S, N, 2)),
                              visibs=torch.zeros((S, N)), valids=torch.zeros((S, N)))
