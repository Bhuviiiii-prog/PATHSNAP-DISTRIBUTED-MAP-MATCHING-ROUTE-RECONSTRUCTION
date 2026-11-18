# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS points to the road arcs a vehicle actually drove, then reconstructs one connected route with an HMM and Viterbi decoding. A FastAPI backend (Laptop 1, macOS) serves predictions to a Streamlit UI (Laptop 2, Linux).

UE24CS352A Machine Learning hackathon, PES University. AI assistance was used and is logged in `AI_USAGE.md`.

## How it works

1. **Validation and EDA:** trip inventory, timestamp and gap checks, road-network checks (`eda/`).
2. **Projection:** each trip is projected to metres with an azimuthal-equidistant projection centred on the trip (`src/geo.py`).
3. **Labels:** each GPS point is matched to an arc of its own reference route, in driving order (`src/labels.py`).
4. **Candidates and features:** 8 nearest arcs per point, 9 features each (`src/features.py`).
5. **Models:** 5 models from different families, 3-fold GroupKFold by trip (`modelling/`).
6. **Tuning:** Optuna TPE on HistGradientBoosting (`tuning/`).
7. **HMM/Viterbi:** emissions from the classifier, transitions from road connectivity (`src/hmm.py`).
8. **API and UI:** `api/app.py`, `ui/app.py`.

## Results (student_test, 10 trips, 19,160 GPS points)

See `experiments/` (exact numbers come from the latest run). Top-1 accuracy: nearest road about 54%, tuned HistGradientBoosting about 86%, plus HMM about 88%. Route continuity rises from about 94% to about 98% with the HMM.

## Install (Laptop 1, macOS/Linux, Python 3.9–3.12)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the dataset at `data/raw/Student folder/` (with `train/` and `student_test/`).

## Run

```bash
python run_data.py      # Part 1: validation, pre-processing, features
python run_models.py    # Part 2: 5 models, selection, Optuna tuning
python run_hmm.py       # Part 3: classifier-only vs HMM/Viterbi
python run_eval.py --dataset evaluator_test   # score a dataset with the saved model
```

## Deploy (two laptops)

```bash
# Laptop 1
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Laptop 2 (Linux)
python3 -m venv .venv && source .venv/bin/activate
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Full step-by-step instructions: `DEMO_SETUP.md`.

## Folder layout

| Folder | Contents |
| --- | --- |
| `src/` | Data loading, projection, labels, features, HMM, prediction pipeline |
| `eda/` | Validation script, trip inventory, Folium map, EDA charts |
| `features/` | Builds the training and test tables |
| `modelling/` | Model comparison and metrics |
| `tuning/` | Optuna search, tuned vs untuned, confusion matrix |
| `api/`, `ui/` | FastAPI backend and Streamlit UI |
| `experiments/`, `figures/`, `maps/`, `models/` | Results, charts, maps, saved model |
