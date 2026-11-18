# AI usage log (for the report's disclosure section)

Tools: Claude Code (Claude Sonnet 5.5) in Session 1; Claude chat (Claude Opus 5.5) in Sessions 2-3 and for the study guides.

## Phase 1
- Read the brief and the PDFs, ran quick data checks (scratch, not committed), and found the global trip spread, the duplicate arcs and the offset trips 33/54.
- Wrote `src/data_io.py` (file loading), `src/geo.py` (per-trip projection, point–segment distance, bearings) and `src/labels.py` (nearest-arc and monotonic DP alignment; the DP is not used yet).
- Wrote and ran `eda/run_eda.py` (inventory, gap stats, network size/components, Folium map).
- Wrote and ran `features/build_features.py` (labels, K-nearest candidates, 6 features).
- Wrote and ran `modelling/common.py` and `modelling/train_compare.py` (5 models, GroupKFold CV, test metrics, plot).
- Wrote and ran `tuning/tune_optuna.py` (15-trial TPE, tuned vs untuned, saved model).
- Wrote PHASE1_STATUS.md, DECISIONS.md and this file.
- The plan was cut short to a minimal version 15 min before evaluation, on the team's instruction.
- All numbers come from the script outputs in `experiments/` and `eda/`.

- Extra plots (Claude, chat session): `eda/extra_plots.py` reads the saved CSV/JSON results (no retraining) and writes `figures/eda_gps_points_per_trip.png`, `figures/eda_max_time_gap_per_trip.png` and `figures/optuna_history.png`.

- `run_phase1.sh` (Claude, chat session): one script that runs all Phase 1 steps in order.

## Session 2 (Claude, chat session, 12:10-13:00)
- Phase 1 polish: `run_data.py`, `run_models.py`, `src/pretty.py` (readable step-by-step output), `src/features.py` (candidate/feature code shared with the API), route continuity and confusion matrix added to the metrics.
- Phase 2: `src/hmm.py` (HMM transitions + Viterbi), `src/pipeline.py`, `run_hmm.py`, `api/app.py` (FastAPI), `ui/app.py` (Streamlit).
- Docs: `README.md`, `DEMO_SETUP.md`, `COMMIT_PLAN.md`.
- All results are produced by running these scripts; nothing was typed in by hand.

## Session 3 (Claude, chat session, 12:50-13:05)
- Tested 3 improvements in a sandbox and applied them: order-preserving labels, wider heading window, 3 neighbour features; tie-break rule for model choice. See DECISIONS.md.

## Session 4 (Claude, chat session, 14:00)
- Error analysis of HMM output and the U-turn penalty in src/hmm.py / run_hmm.py.

## Session 5 (Claude, chat session, 14:15)
- Dataset switch: src/datasets.py, API serves student_test and evaluator_test, UI sidebar switch, run_eval.py (scores a dataset with the saved model; no retraining). No model, feature or HMM setting was changed after seeing evaluator results.

## Files mentioned above that are not in the repository
- `PATHSNAP_BRIEF.md` (the written instructions given to Claude Code), `PHASE1_STATUS.md` (status of the first quick version), `COMMIT_PLAN.md` and `run_phase1.sh` were working files, replaced by `README.md`, `DEMO_SETUP.md` and the `run_*.py` scripts. They are kept by the team but not committed.
