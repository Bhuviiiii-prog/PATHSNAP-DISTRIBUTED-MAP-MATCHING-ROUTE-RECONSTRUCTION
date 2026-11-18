"""Extra report plots made from the CSV/JSON files the pipeline already wrote.
Nothing is retrained here; it only reads saved results.

Output (in figures/):
  eda_gps_points_per_trip.png - how long each trip is
  eda_max_time_gap_per_trip.png - biggest GPS gap in each trip (log scale)
  optuna_history.png - CV macro F1 of each Optuna trial and the best so far

Run from the project folder: .venv/bin/python eda/extra_plots.py
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project folder = one level above this file (can be overridden for testing)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

inv = pd.read_csv(os.path.join(ROOT, "eda", "trip_inventory.csv"), dtype={"trip": str})
colours = {"train": "tab:blue", "student_test": "tab:orange"}

# 1) GPS points per trip, sorted, coloured by partition
inv = inv.sort_values("gps_rows").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(inv.index, inv.gps_rows, color=inv.partition.map(colours))
ax.set_yscale("log")
ax.set_xlabel("trip (sorted by length)")
ax.set_ylabel("GPS points (log scale)")
ax.set_title("GPS points per trip: train (blue) vs student_test (orange)")
ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_gps_points_per_trip.png"), dpi=120)
plt.close()

# 2) biggest time gap per trip; anything above 10 s is a real GPS dropout
inv = inv.sort_values("dt_max_s").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(inv.index, inv.dt_max_s, color=inv.partition.map(colours))
ax.axhline(10, color="grey", linestyle="--", linewidth=1)
ax.text(1, 12, "10 s", color="grey")
ax.set_yscale("log")
ax.set_xlabel("trip (sorted by biggest gap)")
ax.set_ylabel("biggest gap between GPS points (s, log scale)")
ax.set_title("Largest time gap per trip: train (blue) vs student_test (orange)")
ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_max_time_gap_per_trip.png"), dpi=120)
plt.close()

# 3) Optuna history: score of every trial and the running best
trials = pd.read_csv(os.path.join(ROOT, "experiments", "optuna_trials.csv"))
trials = trials[trials.state == "COMPLETE"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trials.number, trials.value, "o", label="trial")
ax.plot(trials.number, trials.value.cummax(), "-", label="best so far")
ax.set_xlabel("Optuna trial")
ax.set_ylabel("CV macro F1 (3-fold GroupKFold, train)")
ax.set_title("Bayesian optimisation (TPE) of HistGradientBoosting")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "optuna_history.png"), dpi=120)
plt.close()

print("saved 3 plots to", OUT)
