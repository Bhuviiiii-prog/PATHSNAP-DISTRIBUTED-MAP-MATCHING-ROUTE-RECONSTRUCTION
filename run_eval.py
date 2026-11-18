"""Score a whole dataset with the SAVED model and SAVED HMM settings (nothing is retrained or re-tuned).

Run:  python run_eval.py --dataset student_test
      python run_eval.py --dataset evaluator_test
Prints one row per trip and the point-weighted total. Writes nothing to disk.
"""
import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import os, sys, json, argparse
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.pretty import banner, step, note, table
from src.datasets import available_datasets, trip_ids
from src.pipeline import load_trip_from_folder, predict_trip

datasets = available_datasets()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=list(datasets), default="student_test")
args = ap.parse_args()
folder = datasets[args.dataset]

model = joblib.load(os.path.join(ROOT, "models", "hgb_tuned.joblib"))
params = json.load(open(os.path.join(ROOT, "models", "hmm_params.json")))

banner(f"PATHSNAP  |  EVALUATION ON {args.dataset.upper()}")
note(f"folder: {folder}")
note(f"saved model: models/hgb_tuned.joblib | saved HMM settings: {params}")

step(1, "Classifier-only vs HMM, trip by trip")
rows = []
for t in trip_ids(folder):
    r = predict_trip(load_trip_from_folder(folder, t), model, params)
    rows.append({"trip": t, "points": r["n_points"],
                 "top1_classifier": r.get("top1_clf"), "top1_hmm": r.get("top1_hmm"),
                 "continuity_classifier": r["continuity_clf"], "continuity_hmm": r["continuity_hmm"],
                 "seconds": r["seconds"]["total"]})
res = pd.DataFrame(rows)
has_truth = res.top1_hmm.notna().all()
w = res.points / res.points.sum()
total = {"trip": "ALL (point-weighted)", "points": int(res.points.sum()), "seconds": float(res.seconds.sum())}
for c in ["top1_classifier", "top1_hmm", "continuity_classifier", "continuity_hmm"]:
    total[c] = float((res[c] * w).sum()) if res[c].notna().all() else None
res = pd.concat([res, pd.DataFrame([total])], ignore_index=True)
table(res)
if has_truth:
    note(f"overall Top-1 {total['top1_classifier']:.4f} -> {total['top1_hmm']:.4f} with HMM, "
         f"continuity {total['continuity_classifier']:.4f} -> {total['continuity_hmm']:.4f}")
else:
    note("no .route files in this folder, so accuracy cannot be computed (routes are still predicted)")
banner("DONE (nothing was saved or changed)")
