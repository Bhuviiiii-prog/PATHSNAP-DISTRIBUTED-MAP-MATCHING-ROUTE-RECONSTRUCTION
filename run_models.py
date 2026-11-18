"""PART 2 - MODELS: train and compare 5 models, choose one, tune it with Optuna.

Run:  .venv/bin/python run_models.py            (about 2-3 min)
      .venv/bin/python run_models.py --trials 5  (quicker tuning for a live demo)
Needs run_data.py to have been run first (it builds data/processed/).
"""
import warnings
# Hide harmless library warnings so the demo output stays readable:
# - urllib3 complains about the macOS system SSL library (no effect on us)
# - numpy 2.0 + Apple's maths library print false "overflow in matmul" warnings
#   during Logistic Regression; the scores are unaffected (checked against Linux runs)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import os, sys, argparse
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modelling"))
sys.path.insert(0, os.path.join(ROOT, "tuning"))
from src.pretty import banner, step, note, table, Timer
from common import FEATURES, TRAIN_POINT_STRIDE, load, nearest_road_top1
from train_compare import MODELS, run_one, save
from tune_optuna import run_search, compare, confusion
from eda.extra_plots import plot_optuna

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=15, help="number of Optuna trials")
args = ap.parse_args()

banner("PATHSNAP  |  PART 2: MODEL TRAINING, SELECTION AND TUNING")

# ---------------------------------------------------------------------------
step(1, "Load the feature tables built in Part 1")
train, test = load("train"), load("student_test")
note(f"train       : {train.trip.nunique()} trips, {len(train):,} candidate rows (every {TRAIN_POINT_STRIDE}th GPS point)")
note(f"student_test: {test.trip.nunique()} trips, {len(test):,} candidate rows (every GPS point)")
note(f"features    : {', '.join(FEATURES)}")
note("validation  : 3-fold GroupKFold by trip on train (no trip is in two folds)")

# ---------------------------------------------------------------------------
step(2, "Baseline with no learning: always pick the nearest road")
base = nearest_road_top1(test)
note(f"student_test Top-1 = {base:.4f}  ({100 * base:.1f}% of GPS points matched correctly)")

# ---------------------------------------------------------------------------
step(3, "Train and compare 5 models from different families")
rows = []
for name in MODELS:
    with Timer(f"{name}") as tm:
        r = run_one(name, train, test)
    rows.append(r)
    note(f"   CV macro F1 {r['cv_macro_f1']:.4f}  |  test Top-1 {r['top1_acc']:.4f}  |  log loss {r['log_loss']:.4f}")
df = save(rows)
show = df[["model", "family", "cv_macro_f1", "cv_macro_f1_std", "top1_acc", "macro_f1", "macro_precision",
           "macro_recall", "log_loss", "route_continuity", "train_s", "inference_s"]]
print()
table(show)

# ---------------------------------------------------------------------------
step(4, "Choose the model (on CV macro F1 from train trips, never on test)")
# Rule: highest CV macro F1. Differences smaller than TIE (far smaller than the fold-to-fold
# std, about 0.013 with 9 features) are treated as a tie, broken by faster inference: the API must
# answer live, and a faster, smaller model is easier to deploy.
TIE = 0.002
top = df[df.cv_macro_f1 >= df.cv_macro_f1.max() - TIE]
best = top.sort_values(["inference_s", "log_loss"]).iloc[0]
if len(top) > 1:
    note(f"tie on CV macro F1 (within {TIE}): {', '.join(top.model)} -> broken by inference speed")
note(f"chosen: {best.model}  (CV macro F1 {best.cv_macro_f1:.4f}, log loss {best.log_loss:.4f})")
note(f"it also has test Top-1 {best.top1_acc:.4f} vs {base:.4f} for nearest road, and log loss {best.log_loss:.4f}")

# ---------------------------------------------------------------------------
step(5, f"Bayesian hyperparameter optimisation (Optuna TPE, {args.trials} trials, train CV only)")


def on_trial(study, trial):
    flag = "  <- best so far" if trial.number == study.best_trial.number else ""
    note(f"trial {trial.number:2d}: CV macro F1 {trial.value:.4f}{flag}")


with Timer("search"):
    study, _ = run_search(train, args.trials, on_trial)
note("best settings: " + ", ".join(f"{k}={v:.4g}" for k, v in study.best_params.items()))

# ---------------------------------------------------------------------------
step(6, "Tuned vs untuned (same data, same scores)")
cmp_df, model = compare(train, test, study.best_params)
table(cmp_df[["model", "cv_macro_f1", "top1_acc", "macro_f1", "log_loss", "route_continuity", "inference_s"]])
cm = confusion(model, test)
note("confusion matrix (student_test candidate rows, top-1 rule):")
table(pd.DataFrame(cm, index=["true: wrong arc", "true: right arc"], columns=["pred: wrong", "pred: right"]),
      "{:d}", index=True)
plot_optuna(ROOT)

banner("PART 2 DONE: model saved to models/hgb_tuned.joblib")
note("pictures: experiments/model_comparison.png, experiments/confusion_matrix.png, figures/optuna_history.png")
note("results : experiments/model_comparison.csv, tuned_vs_untuned.csv, optuna_trials.csv")
