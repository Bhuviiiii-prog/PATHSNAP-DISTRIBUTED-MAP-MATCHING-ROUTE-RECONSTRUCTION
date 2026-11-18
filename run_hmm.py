"""PART 3 - ROUTE RECONSTRUCTION: classifier-only vs classifier + HMM/Viterbi.

Run:  .venv/bin/python run_hmm.py      (about 1 min; needs models/hgb_tuned.joblib from run_models.py)
Decoder settings are chosen on a few TRAIN trips, then reported on student_test.
Writes experiments/hmm_settings.csv, experiments/hmm_comparison.csv, maps/student_test_00000000.html
"""
import warnings
# Hide harmless library warnings so the demo output stays readable:
# - urllib3 complains about the macOS system SSL library (no effect on us)
# - numpy 2.0 + Apple's maths library print false "overflow in matmul" warnings
#   during Logistic Regression; the scores are unaffected (checked against Linux runs)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import os, sys, json, itertools
import joblib
import numpy as np
import pandas as pd
import folium

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.pretty import banner, step, note, table, Timer
from src.data_io import list_trips, load_trip
from src.pipeline import predict_trip, arcs_to_lines

TUNE_TRIPS = 10                         # train trips used to pick decoder settings
GRID = {"HOP": [1.0, 2.0], "FAR": [10.0, 20.0], "UTURN": [5.0, 10.0, 20.0]}

banner("PATHSNAP  |  PART 3: HMM + VITERBI ROUTE RECONSTRUCTION")
model = joblib.load(os.path.join(ROOT, "models", "hgb_tuned.joblib"))
os.makedirs(os.path.join(ROOT, "maps"), exist_ok=True)

# ---------------------------------------------------------------------------
step(1, f"Choose decoder settings on {TUNE_TRIPS} train trips")
dropped = json.load(open(os.path.join(ROOT, "experiments", "features_report.json")))["train"]["dropped_trips"]
tune_ids = [t for t in list_trips("train") if t not in dropped][:TUNE_TRIPS]
settings = []
with Timer("grid search"):
    for hop, far, ut in itertools.product(GRID["HOP"], GRID["FAR"], GRID["UTURN"]):
        res = [predict_trip(load_trip("train", t), model, {"HOP": hop, "FAR": far, "UTURN": ut}) for t in tune_ids]
        pts = sum(r["n_points"] for r in res)
        settings.append({"HOP": hop, "FAR": far, "UTURN": ut,
                         "top1_hmm": sum(r["top1_hmm"] * r["n_points"] for r in res) / pts,
                         "continuity_hmm": sum(r["continuity_hmm"] * r["n_points"] for r in res) / pts})
settings = pd.DataFrame(settings).sort_values("top1_hmm", ascending=False)
settings.to_csv(os.path.join(ROOT, "experiments", "hmm_settings.csv"), index=False)
table(settings)
best = {"HOP": float(settings.iloc[0].HOP), "FAR": float(settings.iloc[0].FAR), "UTURN": float(settings.iloc[0].UTURN)}
note(f"chosen: HOP={best['HOP']}, FAR={best['FAR']}, UTURN={best['UTURN']}")
json.dump(best, open(os.path.join(ROOT, "models", "hmm_params.json"), "w"))

# ---------------------------------------------------------------------------
step(2, "Classifier-only vs HMM on every student_test trip")
rows = []
for t in list_trips("student_test"):
    r = predict_trip(load_trip("student_test", t), model, best)
    rows.append({"trip": t, "points": r["n_points"], "top1_classifier": r["top1_clf"], "top1_hmm": r["top1_hmm"],
                 "continuity_classifier": r["continuity_clf"], "continuity_hmm": r["continuity_hmm"],
                 "seconds": r["seconds"]["total"]})
res = pd.DataFrame(rows)
w = res.points / res.points.sum()
total = {"trip": "ALL (point-weighted)", "points": int(res.points.sum()),
         **{c: float((res[c] * w).sum()) for c in ["top1_classifier", "top1_hmm", "continuity_classifier", "continuity_hmm"]},
         "seconds": float(res.seconds.sum())}
res = pd.concat([res, pd.DataFrame([total])], ignore_index=True)
res.to_csv(os.path.join(ROOT, "experiments", "hmm_comparison.csv"), index=False)
table(res)
better = (res.top1_hmm[:-1] > res.top1_classifier[:-1]).sum()
note(f"HMM improved Top-1 on {better} of {len(res) - 1} trips")
note(f"overall Top-1 {total['top1_classifier']:.4f} -> {total['top1_hmm']:.4f}, "
     f"continuity {total['continuity_classifier']:.4f} -> {total['continuity_hmm']:.4f}")

# ---------------------------------------------------------------------------
step(3, "Map for the report: GPS, reference route, classifier-only, HMM")
trip = load_trip("student_test", "00000000")
r = predict_trip(trip, model, best)
tr = trip["track"]
m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14)
layers = [("Reference route (green)", trip["route"], "green", 6),
          ("Classifier only (orange)", r["clf_arcs"], "orange", 3),
          ("HMM / Viterbi (red)", r["hmm_arcs"], "red", 3)]
for title, seq, colour, weight in layers:
    fg = folium.FeatureGroup(title)
    for line in arcs_to_lines(np.asarray(seq), trip):
        folium.PolyLine(line, color=colour, weight=weight, opacity=0.8).add_to(fg)
    fg.add_to(m)
gps = folium.FeatureGroup("GPS points (blue)")
for lat, lon in zip(tr.lat, tr.lon):
    folium.CircleMarker((lat, lon), radius=1.5, color="blue").add_to(gps)
gps.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.save(os.path.join(ROOT, "maps", "student_test_00000000.html"))
note("saved maps/student_test_00000000.html")

banner("PART 3 DONE: decoder settings saved to models/hmm_params.json")
