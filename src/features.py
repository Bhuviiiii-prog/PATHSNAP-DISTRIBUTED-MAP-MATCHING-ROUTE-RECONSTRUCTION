"""Candidates and features for one trip (no labels needed).

Shared by the training pipeline (features/build_features.py) and by the API,
which must build exactly the same features for a new trip without its route.

For every GPS point:
  1. pre-select the PREFILTER arcs whose middle point is closest (KD-tree, fast)
  2. measure the exact point-to-segment distance to those arcs
  3. keep the K closest arcs = the candidates
Each (point, candidate) pair becomes one row with 9 features.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import arc_endpoints, point_segment, bearing_deg, angle_diff

K = 8           # candidates kept per point
PREFILTER = 40  # arcs pre-selected by midpoint distance before the exact distance
HEADING_WINDOW = 3  # heading/speed measured from point i-3 to i+3 (less noisy when slow)
NEIGHBOURS = (-2, -1, 1, 2)  # nearby GPS points used for the neighbour features
FEATURES = ["dist", "dist_rank", "bearing_diff", "seg_len", "speed", "n_cand_30m",
            "nb_dist_mean", "nb_dist_max", "nb_rank"]


def gps_heading_and_speed(track, w=HEADING_WINDOW):
    """Heading (degrees) and speed (m/s) at every point, measured from point i-w to i+w.
    A wider window is less noisy: at low speed, a 1 s step is mostly GPS jitter, which
    made the direction feature unreliable on two-way roads (tested: w=1 -> w=3 helped)."""
    x, y, t = track.x.to_numpy(), track.y.to_numpy(), track.t.to_numpy()
    i = np.arange(len(x))
    a, b = np.clip(i - w, 0, len(x) - 1), np.clip(i + w, 0, len(x) - 1)
    heading = bearing_deg(x[b] - x[a], y[b] - y[a])
    speed = np.hypot(x[b] - x[a], y[b] - y[a]) / np.maximum(t[b] - t[a], 1e-9)
    return heading, speed


def candidate_features(trip):
    """Returns (rows, cand): a DataFrame with one row per (point, candidate),
    and the (n_points, K) array of candidate arc ids. The trip must already
    be projected to metres (geo.project_trip)."""
    tr = trip["track"]
    ax, ay, bx, by = arc_endpoints(trip)
    mid = cKDTree(np.c_[(ax + bx) / 2, (ay + by) / 2])
    px, py = tr.x.to_numpy(), tr.y.to_numpy()
    _, pre = mid.query(np.c_[px, py], k=min(PREFILTER, len(ax)))
    dist, _ = point_segment(px[:, None], py[:, None], ax[pre], ay[pre], bx[pre], by[pre])
    order = np.argsort(dist, axis=1)[:, :K]
    cand = np.take_along_axis(pre, order, 1)
    cdist = np.take_along_axis(dist, order, 1)

    heading, speed = gps_heading_and_speed(tr)

    # Neighbour features: is the same candidate arc also close to the nearby GPS points?
    # A road that stays close for several seconds is more likely than one that is close once.
    n = len(cand)
    idx = np.arange(n)
    nb = []
    for off in NEIGHBOURS:
        j = np.clip(idx + off, 0, n - 1)
        d, _ = point_segment(px[j][:, None], py[j][:, None], ax[cand], ay[cand], bx[cand], by[cand])
        nb.append(d)
    nb = np.stack(nb)                                        # (neighbours, n, K)
    nb_mean = nb.mean(0)
    nb_rank = np.argsort(np.argsort(nb_mean + cdist, 1), 1)  # 0 = best over the whole window
    arc_bear = bearing_deg(bx - ax, by - ay)
    seg_len = np.hypot(bx - ax, by - ay)
    n, k = cand.shape
    rows = pd.DataFrame({
        "trip": trip["id"], "point": np.repeat(np.arange(n), k), "arc": cand.ravel(),
        "dist": cdist.ravel(), "dist_rank": np.tile(np.arange(k), n),
        "bearing_diff": angle_diff(np.repeat(heading, k), arc_bear[cand.ravel()]),
        "seg_len": seg_len[cand.ravel()], "speed": np.repeat(speed, k),
        "n_cand_30m": np.repeat((cdist <= 30).sum(1), k),
        "nb_dist_mean": nb_mean.ravel(), "nb_dist_max": nb.max(0).ravel(), "nb_rank": nb_rank.ravel(),
    })
    return rows, cand
