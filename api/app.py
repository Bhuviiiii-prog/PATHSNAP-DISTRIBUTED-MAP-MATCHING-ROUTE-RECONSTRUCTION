"""PathSnap API (runs on Laptop 1, the MacBook; the UI runs on Laptop 2, a Linux laptop).

Start:  uvicorn api.app:app --host 0.0.0.0 --port 8000

Two datasets can be served at the same time; the UI chooses one per request:
  student_test    data/raw/Student folder/student_test
  evaluator_test  data/raw/evaluator_test  (or the folder in PATHSNAP_EVAL_DIR)
A dataset whose folder does not exist is simply not offered.

Endpoints
  GET  /health                     -> server up, datasets and trip counts
  GET  /datasets                   -> names of the available datasets
  GET  /trips?dataset=NAME         -> trip ids of that dataset
  POST /predict                    -> {"trip_id": "...", "dataset": "..."} -> predicted route + accuracy

The ground-truth route never leaves this server: the UI only receives the
predicted route, the GPS points and the accuracy numbers.
"""
import warnings
# Hide harmless library warnings so the demo output stays readable:
# - urllib3 complains about the macOS system SSL library (no effect on us)
# - numpy 2.0 + Apple's maths library print false "overflow in matmul" warnings
#   during Logistic Regression; the scores are unaffected (checked against Linux runs)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import os, sys, json, time, logging
from typing import Optional
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pipeline import load_trip_from_folder, predict_trip, arcs_to_lines
from src.datasets import available_datasets, trip_ids, DEFAULT_ORDER

DATASETS = available_datasets()
DEFAULT = next((d for d in DEFAULT_ORDER if d in DATASETS), None)
MODEL = joblib.load(os.path.join(ROOT, "models", "hgb_tuned.joblib"))
HMM_PARAMS = json.load(open(os.path.join(ROOT, "models", "hmm_params.json")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")
app = FastAPI(title="PathSnap API")


def folder_of(dataset):
    name = dataset or DEFAULT
    if name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {name}; available: {list(DATASETS)}")
    return name, DATASETS[name]


@app.on_event("startup")
def show_config():
    log.info(f"model loaded | HMM settings {HMM_PARAMS}")
    for name, folder in DATASETS.items():
        log.info(f"dataset {name:15s} {len(trip_ids(folder)):3d} trips | {folder}")


@app.get("/health")
def health():
    return {"status": "ok", "default_dataset": DEFAULT,
            "datasets": {n: len(trip_ids(f)) for n, f in DATASETS.items()}}


@app.get("/datasets")
def datasets():
    return {"datasets": list(DATASETS), "default": DEFAULT}


@app.get("/trips")
def trips(dataset: Optional[str] = None):
    name, folder = folder_of(dataset)
    return {"dataset": name, "trips": trip_ids(folder)}


class PredictRequest(BaseModel):
    trip_id: str
    dataset: Optional[str] = None


@app.post("/predict")
def predict(req: PredictRequest):
    name, folder = folder_of(req.dataset)
    if req.trip_id not in trip_ids(folder):
        raise HTTPException(404, f"unknown trip {req.trip_id} in {name}")
    log.info(f"REQUEST  dataset={name} trip={req.trip_id}")
    t0 = time.time()
    trip = load_trip_from_folder(folder, req.trip_id)
    r = predict_trip(trip, MODEL, HMM_PARAMS)
    tr = trip["track"]
    resp = {
        "dataset": name,
        "trip_id": req.trip_id,
        "n_points": r["n_points"],
        "gps": [[float(a), float(b)] for a, b in zip(tr.lat, tr.lon)],
        "route_hmm": arcs_to_lines(r["hmm_arcs"], trip),
        "route_classifier": arcs_to_lines(r["clf_arcs"], trip),
        "metrics": {
            "top1_hmm": r.get("top1_hmm"), "top1_classifier": r.get("top1_clf"),
            "continuity_hmm": r["continuity_hmm"], "continuity_classifier": r["continuity_clf"],
        },
        "seconds": {k: round(v, 3) for k, v in r["seconds"].items()},
    }
    acc = "n/a" if r.get("top1_hmm") is None else f"{r['top1_hmm']:.4f}"
    log.info(f"RESPONSE dataset={name} trip={req.trip_id} points={r['n_points']} top1_hmm={acc} "
             f"continuity_hmm={r['continuity_hmm']:.4f} time={time.time() - t0:.2f}s")
    return resp
