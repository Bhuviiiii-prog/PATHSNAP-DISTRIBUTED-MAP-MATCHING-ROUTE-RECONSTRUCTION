"""Extra report plots made from the CSV/JSON files the pipeline already wrote.
Nothing is retrained here; it only reads saved results.

Output (in figures/):
  eda_gps_points_per_trip.png   - how long each trip is
  eda_max_time_gap_per_trip.png - biggest GPS gap in each trip (log scale)
  optuna_history.png            - CV macro F1 of each Optuna trial and the best so far

Run on its own: .venv/bin/python eda/extra_plots.py
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLOURS = {"train": "tab:blue", "student_test": "tab:orange"}


def plot_eda(root=ROOT):
    out = os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    inv = pd.read_csv(os.path.join(root, "eda", "trip_inventory.csv"), dtype={"trip": str})

    # 1) GPS points per trip, sorted, coloured by partition
    inv = inv.sort_values("gps_rows").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(inv.index, inv.gps_rows, color=inv.partition.map(COLOURS))
    ax.set_yscale("log")
    ax.set_xlabel("trip (sorted by length)")
    ax.set_ylabel("GPS points (log scale)")
    ax.set_title("GPS points per trip: train (blue) vs student_test (orange)")
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_gps_points_per_trip.png"), dpi=120)
    plt.close()

    # 2) biggest time gap per trip; anything above 10 s is a real GPS dropout
    inv = inv.sort_values("dt_max_s").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(inv.index, inv.dt_max_s, color=inv.partition.map(COLOURS))
    ax.axhline(10, color="grey", linestyle="--", linewidth=1)
    ax.text(1, 12, "10 s", color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("trip (sorted by biggest gap)")
    ax.set_ylabel("biggest gap between GPS points (s, log scale)")
    ax.set_title("Largest time gap per trip: train (blue) vs student_test (orange)")
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_max_time_gap_per_trip.png"), dpi=120)
    plt.close()


def plot_optuna(root=ROOT):
    path = os.path.join(root, "experiments", "optuna_trials.csv")
    if not os.path.exists(path):
        print("no optuna_trials.csv yet, skipping optuna plot")
        return
    trials = pd.read_csv(path)
    trials = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trials.number, trials.value, "o", label="trial")
    ax.plot(trials.number, trials.value.cummax(), "-", label="best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("CV macro F1 (3-fold GroupKFold, train)")
    ax.set_title("Bayesian optimisation (TPE) of HistGradientBoosting")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(root, "figures", "optuna_history.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    plot_eda(root)
    plot_optuna(root)
    print("saved plots to", os.path.join(root, "figures"))
