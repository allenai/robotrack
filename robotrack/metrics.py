"""TAP-Vid metrics (Doersch et al., 2022): average Jaccard (AJ), position accuracy (<δ^x_avg) and occlusion accuracy (OA).

Adapted from the reference implementation in https://github.com/google-deepmind/tapnet (Apache 2.0), as used by
AllTracker and CoTracker. Coordinates are scaled to a 256x256 frame before thresholding, as in TAP-Vid.
"""
import numpy as np

THRESHOLDS = (1, 2, 4, 8, 16)


def tapvid_metrics(query_frames, gt_occluded, gt_tracks, pred_occluded, pred_tracks, image_size):
    """Metrics for one clip, "first" query mode: each point is scored on the frames after its query frame.

    query_frames  [N]        int, frame each point is queried at (its first visible frame)
    gt_occluded   [N, T]     bool
    gt_tracks     [N, T, 2]  x, y in pixels of an image of `image_size` (H, W)
    pred_occluded [N, T]     bool
    pred_tracks   [N, T, 2]
    Returns a dict of floats in [0, 1].
    """
    T = gt_tracks.shape[1]
    after_query = np.arange(T)[None, :] > np.asarray(query_frames)[:, None]   # [N, T]

    visible, pred_visible = ~gt_occluded, ~pred_occluded
    out = {"oa": np.sum((pred_occluded == gt_occluded) & after_query) / np.sum(after_query)}

    H, W = image_size
    scale = np.array([(W - 1) / 255.0, (H - 1) / 255.0])
    dist2 = np.sum(np.square(pred_tracks / scale - gt_tracks / scale), axis=-1)
    n_visible = np.sum(visible & after_query)
    for thr in THRESHOLDS:
        within = dist2 < thr ** 2
        correct = within & visible
        out[f"d_{thr}"] = np.sum(correct & after_query) / n_visible
        true_pos = np.sum(correct & pred_visible & after_query)
        false_pos = np.sum(((~visible) | (~within)) & pred_visible & after_query)
        out[f"jac_{thr}"] = true_pos / (n_visible + false_pos)
    out["d_avg"] = np.mean([out[f"d_{t}"] for t in THRESHOLDS])
    out["aj"] = np.mean([out[f"jac_{t}"] for t in THRESHOLDS])
    return {k: float(v) for k, v in out.items()}
