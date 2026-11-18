"""Labels, candidates and features -> data/processed/{train,student_test}.parquet

Label: each GPS point gets the nearest arc from its own trip's .route list
       (simple version; the monotonic alignment in src/labels.py is for after lunch).
Candidates: the K nearest arcs to each point (by point-to-segment distance).
One row per (point, candidate) with y = 1 if the candidate is the label arc.
Run: .venv/bin/python features/build_features.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_io import PARTITIONS, list_trips, load_trip
from src.geo import project_trip, arc_endpoints, point_segment, bearing_deg, angle_diff
from src.labels import route_distance_matrix, nearest_route_arc

K = 8          # candidates kept per point
PREFILTER = 40  # arcs pre-selected by midpoint distance before exact distance
# Train trips whose GPS sits >50 m (median) from their own route have unreliable
# labels (EDA found trips 33 and 54 offset by 120-140 m), so they are dropped from train.
MAX_MEDIAN_LABEL_DIST = 50.0


def gps_heading_and_speed(track):
    x, y, t = track.x.to_numpy(), track.y.to_numpy(), track.t.to_numpy()
    # central difference (i-1 -> i+1) is less noisy than a single step
    xp, yp, tp = np.r_[x[0], x[:-1]], np.r_[y[0], y[:-1]], np.r_[t[0], t[:-1]]
    xn, yn, tn = np.r_[x[1:], x[-1]], np.r_[y[1:], y[-1]], np.r_[t[1:], t[-1]]
    heading = bearing_deg(xn - xp, yn - yp)
    speed = np.hypot(xn - xp, yn - yp) / np.maximum(tn - tp, 1e-9)
    return heading, speed


def trip_rows(trip):
    project_trip(trip)
    tr = trip["track"]
    D = route_distance_matrix(trip)
    label = trip["route"][nearest_route_arc(D)]
    label_dist = D.min(axis=1)

    ax, ay, bx, by = arc_endpoints(trip)
    mid = cKDTree(np.c_[(ax + bx) / 2, (ay + by) / 2])
    px, py = tr.x.to_numpy(), tr.y.to_numpy()
    _, pre = mid.query(np.c_[px, py], k=min(PREFILTER, len(ax)))
    dist, _ = point_segment(px[:, None], py[:, None], ax[pre], ay[pre], bx[pre], by[pre])
    order = np.argsort(dist, axis=1)[:, :K]
    cand = np.take_along_axis(pre, order, 1)
    cdist = np.take_along_axis(dist, order, 1)

    heading, speed = gps_heading_and_speed(tr)
    arc_bear = bearing_deg(bx - ax, by - ay)
    seg_len = np.hypot(bx - ax, by - ay)
    n, k = cand.shape
    return pd.DataFrame({
        "trip": trip["id"], "point": np.repeat(np.arange(n), k), "arc": cand.ravel(),
        "dist": cdist.ravel(), "dist_rank": np.tile(np.arange(k), n),
        "bearing_diff": angle_diff(np.repeat(heading, k), arc_bear[cand.ravel()]),
        "seg_len": seg_len[cand.ravel()], "speed": np.repeat(speed, k),
        "n_cand_30m": np.repeat((cdist <= 30).sum(1), k),
        "y": (cand == label[:, None]).ravel().astype(int),
        "label_dist": np.repeat(label_dist, k),
    })


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for part in PARTITIONS:
        frames, dropped = [], []
        for tid in list_trips(part):
            df = trip_rows(load_trip(part, tid))
            if part == "train" and df.label_dist.median() > MAX_MEDIAN_LABEL_DIST:
                dropped.append(tid)
                continue
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        data.to_parquet(os.path.join(out_dir, f"{part}.parquet"))
        per_point = data.groupby(["trip", "point"]).y.max()
        report[part] = {"points": int(len(per_point)), "rows": int(len(data)), "K": K,
                        "candidate_miss_rate": float(1 - per_point.mean()), "dropped_trips": dropped}
        print(part, report[part])
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "experiments", "features_report.json"), "w"), indent=2)
