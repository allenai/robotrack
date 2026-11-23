"""Point-track sampling and projection for RoboTrack-15M (the "legacy" image sampler).

Ported from the point-tracking utilities RoboTrack added to its MolmoSpaces fork, keeping only what the RoboTrack-15M
recipe uses:
  - candidates are pixels on the robot and the free (movable) objects, sampled from the rendered segmentation and
    depth, unprojected to 3D and stored in the coordinates of the body they lie on;
  - a track's 2D position in any frame is that body-local point moved by the body's pose and projected by the camera;
  - a point is visible when its depth is within 3 cm of the rendered depth at its nearest pixel.
One difference from the fork: when no object pixel is visible at a sampling frame, no candidates are added (the fork
fell back to sampling mesh vertices of every body, background included).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


# ---------------------------------------------------------------- bodies
def object_body_ids(model) -> set[int]:
    """Bodies of free (movable) objects: bodies with a free joint and their descendants."""
    roots = {int(model.jnt_bodyid[j]) for j in np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]}
    out, queue = set(), list(roots)
    while queue:
        bid = queue.pop()
        out.add(bid)
        queue.extend(int(c) for c in np.where(model.body_parentid == bid)[0])
    return out


def robot_body_ids(model, namespace: str = "robot_0/") -> set[int]:
    return {bid for bid in range(model.nbody) if model.body(bid).name.startswith(namespace)}


def trackable_body_ids(model) -> set[int]:
    """Points are sampled on these: free objects and the robot."""
    return object_body_ids(model) | robot_body_ids(model)


# ---------------------------------------------------------------- cameras
def camera_matrices(camera, img_width: int, img_height: int):
    """World-to-camera matrix and pinhole intrinsics from a MolmoSpaces Camera (pose + vertical fov)."""
    world2cam = np.linalg.inv(camera.get_pose())
    fy = (img_height / 2.0) / np.tan(np.radians(camera.fov) / 2.0)
    intrinsics = np.array([[fy, 0, img_width / 2.0], [0, fy, img_height / 2.0], [0, 0, 1]], dtype=np.float32)
    return world2cam, intrinsics


# ---------------------------------------------------------------- sampling
def sample_from_image(model, data, camera, img_width, img_height, depth, seg, max_points, seed, body_ids):
    """Pick up to max_points random pixels whose body is in body_ids; return them in body-local coordinates.

    seg is MolmoSpaces' segmentation render (H, W, 3) with the body id in channel 2.
    Returns local_coords (N, 3), body_ids (N,), world_coords (N, 3).
    """
    rng = np.random.RandomState(seed)
    body_map = seg[:, :, 2]
    ys, xs = np.where(np.isin(body_map, list(body_ids)))
    if len(ys) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros((0, 3), np.float32)
    pick = rng.choice(len(ys), size=min(max_points, len(ys)), replace=False)
    pys, pxs = ys[pick], xs[pick]
    bids = body_map[pys, pxs].astype(np.int32)
    z = depth[pys, pxs].astype(np.float64)

    cam2world = camera.get_pose()
    fy = (img_height / 2.0) / np.tan(np.radians(camera.fov) / 2.0)
    pts_cam = np.stack([(pxs - img_width / 2.0) / fy * z, (pys - img_height / 2.0) / fy * z, z], axis=1)
    world = (pts_cam @ cam2world[:3, :3].T + cam2world[:3, 3]).astype(np.float32)
    local = np.empty_like(world)
    for bid in np.unique(bids):
        m = bids == bid
        local[m] = ((world[m] - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)).astype(np.float32)
    return local, bids, world


# ---------------------------------------------------------------- tracking
def track_frame(xpos, xmat, local_coords, body_ids, world2cam, intrinsics, img_width, img_height, depth,
                occlusion_tolerance=0.03):
    """Project body-local points with the body poses of one frame. Returns coords_2d (N, 2), visibility (N,),
    world points (N, 3). Visible = in frame and not more than occlusion_tolerance metres behind the rendered depth
    at the nearest pixel."""
    n = len(local_coords)
    world = np.empty((n, 3), np.float32)
    for bid in np.unique(body_ids):
        m = body_ids == bid
        world[m] = (local_coords[m] @ xmat[bid].reshape(3, 3).T + xpos[bid]).astype(np.float32)
    cam = (world2cam @ np.hstack([world, np.ones((n, 1), np.float32)]).T).T[:, :3]
    z = cam[:, 2]
    safe = np.where(z < 1e-4, 1e-4, z)
    px = intrinsics[0, 0] * cam[:, 0] / safe + intrinsics[0, 2]
    py = intrinsics[1, 1] * cam[:, 1] / safe + intrinsics[1, 2]
    in_frame = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px < img_width) & (py >= 0) & (py < img_height) & (z > 0)
    vis = np.zeros(n, np.float32)
    idx = np.where(in_frame)[0]
    if len(idx):
        rendered = depth[np.clip(py[idx].astype(int), 0, img_height - 1), np.clip(px[idx].astype(int), 0, img_width - 1)]
        vis[idx] = (z[idx] - rendered < occlusion_tolerance).astype(np.float32)
    return np.stack([px, py], axis=1).astype(np.float32), vis, world


# ---------------------------------------------------------------- output
def save_point_tracks(path: Path, trajs_2d, visibility, points_3d, body_ids, intrinsics, query_frames):
    """One camera of one episode, with the same arrays as the released RoboTrack-15M files."""
    np.savez_compressed(path, trajs_2d=trajs_2d.astype(np.float32), visibility=visibility.astype(np.float32),
                        points_3d=points_3d.astype(np.float32), body_ids=body_ids.astype(np.int32),
                        intrinsics=intrinsics.astype(np.float32), query_frames=np.asarray(query_frames, np.int32),
                        sampling_method=np.asarray("image"), sampling_stride=np.asarray(1, np.int32))
