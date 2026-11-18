"""PART 1 - DATA: validation, preprocessing and feature extraction.

Run:  .venv/bin/python run_data.py
Time: about 20 s.  Writes eda/, figures/eda_*.png, data/processed/, experiments/features_report.json
"""
import warnings
# Hide harmless library warnings so the demo output stays readable:
# - urllib3 complains about the macOS system SSL library (no effect on us)
# - numpy 2.0 + Apple's maths library print false "overflow in matmul" warnings
#   during Logistic Regression; the scores are unaffected (checked against Linux runs)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.pretty import banner, step, note, table, Timer
from src.data_io import PARTITIONS, RAW_DIR, list_trips, load_trip
from src.geo import project_trip
from src.labels import route_distance_matrix, align_monotonic
from src.features import K, FEATURES, candidate_features
from eda.run_eda import trip_stats, make_map
from eda.extra_plots import plot_eda
from features.build_features import build_all

DEMO_PART, DEMO_TRIP, DEMO_POINT = "student_test", "00000000", 116
pd.set_option("display.width", 120)

banner("PATHSNAP  |  PART 1: DATA PRE-PROCESSING AND FEATURE EXTRACTION")

# ---------------------------------------------------------------------------
step(1, "What data do we have?")
for part in PARTITIONS:
    trips = list_trips(part)
    note(f"{part:13s}: {len(trips)} trips, 4 files each (.track .nodes .arcs .route)")
note(f"data folder: {RAW_DIR}")

# ---------------------------------------------------------------------------
step(2, "Data validation (every trip)")
with Timer("validated all trips"):
    inv = pd.DataFrame([trip_stats(load_trip(p, t)) for p in PARTITIONS for t in list_trips(p)])
inv.to_csv(os.path.join(ROOT, "eda", "trip_inventory.csv"), index=False)
summary = inv.groupby("partition").agg(
    trips=("trip", "count"), gps_points_median=("gps_rows", "median"), missing_values=("nan_values", "sum"),
    timestamps_sorted=("time_sorted", "all"), share_steps_1s=("dt_share_1s", "mean"),
    biggest_gap_s=("dt_max_s", "max"), isolated_nodes=("isolated_nodes", "sum"),
    arcs_median=("arcs", "median")).reset_index()
table(summary, "{:.2f}")
note("No missing values, timestamps always increasing, no isolated nodes.")
note("GPS is nominally 1 Hz, but some trips have long dropouts:")
table(inv.sort_values("dt_max_s", ascending=False).head(5)[["partition", "trip", "gps_rows", "dt_max_s", "gaps_over_10s"]], "{:.0f}")

# ---------------------------------------------------------------------------
step(3, f"Pre-processing, shown on one trip ({DEMO_PART} {DEMO_TRIP})")
trip = load_trip(DEMO_PART, DEMO_TRIP)
note("Raw GPS (.track): longitude, latitude, time in seconds")
table(trip["track"].head(3), {"lon": "{:.6f}", "lat": "{:.6f}", "t": "{:.0f}"})
project_trip(trip)
note("After projection to metres (AEQD centred on this trip): x, y")
table(trip["track"].head(3)[["lon", "lat", "t", "x", "y"]], {"lon": "{:.6f}", "lat": "{:.6f}", "t": "{:.0f}", "x": "{:.1f}", "y": "{:.1f}"})
D = route_distance_matrix(trip)
pos = align_monotonic(D)
label = trip["route"][pos]
note("Labels: each GPS point -> an arc of its own route, matched in driving order (never backwards).")
note(f"Median distance from a GPS point to its label arc: {np.median(D[np.arange(len(pos)), pos]):.1f} m")

# ---------------------------------------------------------------------------
step(4, f"Feature extraction: the {K} candidates of GPS point {DEMO_POINT}")
rows, cand = candidate_features(trip)
rows["y"] = (cand == label[:, None]).ravel().astype(int)
table(rows[rows.point == DEMO_POINT][["arc"] + FEATURES + ["y"]], "{:.1f}")
note("Rows 1-2: same road in opposite directions, same distance. Only bearing_diff tells them apart.")
note("nb_* columns: how close the same arc is to the 4 neighbouring GPS points (i-2..i+2).")

# ---------------------------------------------------------------------------
step(5, "Build the training and test tables (all trips)")
with Timer("built features for all trips"):
    report = build_all(verbose=False)
table(pd.DataFrame([{"partition": p, "trips": r["trips"], "gps_points": r["points"], "rows": r["rows"],
                     "candidate_miss_rate_%": 100 * r["candidate_miss_rate"],
                     "dropped_trips": ", ".join(r["dropped_trips"]) or "-"} for p, r in report.items()]), "{:.2f}")
note("Dropped train trips: GPS sits >50 m (median) from their own route, so labels are unreliable.")
train = pd.read_parquet(os.path.join(ROOT, "data", "processed", "train.parquet"))
note("Average feature value, correct candidate (y=1) vs wrong candidates (y=0), train:")
table(train.groupby("y")[FEATURES].mean().reset_index(), "{:.2f}")
note("The correct arc is closer, ranked higher, and points the same way as the car.")

# ---------------------------------------------------------------------------
step(6, "Save pictures")
make_map(load_trip("train", "00000003"), os.path.join(ROOT, "eda", "map_train_00000003.html"))
plot_eda(ROOT)
note("eda/map_train_00000003.html, figures/eda_gps_points_per_trip.png, figures/eda_max_time_gap_per_trip.png")

banner("PART 1 DONE: tables saved in data/processed/  ->  next: python run_models.py")
