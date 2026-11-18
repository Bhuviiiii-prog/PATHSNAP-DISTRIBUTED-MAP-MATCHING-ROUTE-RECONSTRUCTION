"""Labels, candidates and features -> data/processed/{train,student_test}.parquet

Label: each GPS point is matched to an arc of its own trip's .route list with the
       order-preserving alignment in src/labels.py (points can only move forward along the route).
Candidates + features: src/features.py (the API uses the same code).
One row per (point, candidate) with y = 1 if the candidate is the label arc.
Run on its own: .venv/bin/python features/build_features.py
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_io import PARTITIONS, list_trips, load_trip
from src.geo import project_trip
from src.labels import route_distance_matrix, align_monotonic
from src.features import K, candidate_features

# Train trips whose GPS sits >50 m (median) from their own route have unreliable
# labels (EDA found trips 33 and 54 offset by 120-140 m), so they are dropped from train.
MAX_MEDIAN_LABEL_DIST = 50.0


def trip_rows(trip):
    project_trip(trip)
    D = route_distance_matrix(trip)
    pos = align_monotonic(D)
    label = trip["route"][pos]
    label_dist = D[np.arange(len(pos)), pos]
    rows, cand = candidate_features(trip)
    rows["y"] = (cand == label[:, None]).ravel().astype(int)
    rows["label_dist"] = np.repeat(label_dist, K)   # for analysis only, never a feature
    return rows


def build_all(verbose=True):
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
        report[part] = {"trips": len(frames), "points": int(len(per_point)), "rows": int(len(data)), "K": K,
                        "candidate_miss_rate": float(1 - per_point.mean()), "dropped_trips": dropped}
        if verbose:
            print(part, report[part])
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "experiments", "features_report.json"), "w"), indent=2)
    return report


if __name__ == "__main__":
    build_all()
