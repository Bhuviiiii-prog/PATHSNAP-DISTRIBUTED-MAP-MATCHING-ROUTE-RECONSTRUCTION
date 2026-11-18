# PathSnap demo setup: step by step

| Laptop | Machine | Runs |
| --- | --- | --- |
| Laptop 1 | MacBook (`~/pathsnap`) | Data and model scripts, the trained model, the FastAPI server |
| Laptop 2 | Linux laptop | The Streamlit UI only |

Do sections 1 to 4 at least 20 minutes before the evaluation. Both laptops must be on the same Wi-Fi.

## 1. Mac: produce the results (already done once; repeat only if code changes)

```bash
cd ~/pathsnap
source .venv/bin/activate
export LOKY_MAX_CPU_COUNT=10
python run_data.py      # Part 1: validation, pre-processing, features   (~20 s)
python run_models.py    # Part 2: 5 models, selection, Optuna tuning      (~2.5 min)
python run_hmm.py       # Part 3: classifier-only vs HMM/Viterbi          (~30 s)
```

Each script prints numbered steps with tables and ends with a "PART ... DONE" banner.

## 2. Mac: start the API

```bash
cd ~/pathsnap
source .venv/bin/activate
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- If macOS asks "Allow incoming network connections?", click **Allow**.
- You should see `model loaded | HMM settings {...} | 10 trips`.
- **Leave this window open and visible.** Evaluators check the request log here (`REQUEST trip=...` and `RESPONSE trip=... top1_hmm=...`).
- After re-running `run_models.py` or `run_hmm.py`, stop the API (Ctrl+C) and start it again so it loads the new model.

In a second Terminal tab (Cmd+T):

```bash
ipconfig getifaddr en0                 # the Mac's IP address, e.g. 192.168.1.23
curl http://127.0.0.1:8000/health      # should print {"status":"ok", ... "n_trips":10}
```

## 3. Linux laptop: one-time setup (about 10 minutes)

**Get the UI files** (`pathsnap_ui.zip`, 5 KB: `app.py`, `requirements.txt`, this guide). Any one of:

- **Download it** from the shared chat, WhatsApp Web, email or Google Drive. It lands in `~/Downloads`.
- **Copy it from the Mac over Wi-Fi** (this also tests the connection the demo needs). On the Mac: `cd ~/pathsnap/share && python3 -m http.server 9000`. On Linux: `cd ~/Downloads && wget http://<Mac IP>:9000/pathsnap_ui.zip` (or `curl -O ...`). Stop the Mac server with Ctrl+C afterwards.
- **USB stick:** copy `~/pathsnap/share/pathsnap_ui.zip` from the Mac.

**Unzip it** (skip this if the browser already gave you a `pathsnap_ui` folder in Downloads):

```bash
cd ~/Downloads && unzip pathsnap_ui.zip
```

If the file was renamed (e.g. `pathsnap_ui (1).zip`), find it with `ls ~/Downloads | grep pathsnap` and put the exact name in quotes.

**Set up the environment:**

```bash
python3 --version                    # needs 3.9 to 3.12
cd ~/Downloads/pathsnap_ui
ls                                   # must show app.py, requirements.txt, DEMO_SETUP.md
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # 2-4 minutes
```

- If `ls` shows only another `pathsnap_ui` folder, run `cd pathsnap_ui` once more.
- If `python3 -m venv` fails with "ensurepip is not available", run the command the error suggests (for example `sudo apt install -y python3-venv` or `python3.12-venv`), then repeat the last three lines.

## 4. Connect the two laptops

On the Linux laptop:

```bash
curl http://<Mac IP>:8000/health     # must print {"status":"ok", ...}
cd ~/Downloads/pathsnap_ui && source .venv/bin/activate
streamlit run app.py
```

1. A browser opens at `http://localhost:8501` (open it yourself if not; press Enter if Streamlit asks for an email).
2. In the sidebar, replace `http://127.0.0.1:8000` with `http://<Mac IP>:8000`. It should say **"Connected: 10 trips available"**.
3. Pick a trip and click **Reconstruct route**. The accuracy, continuity and map appear on Linux, and the request log appears in the Mac Terminal.

**The Mac IP:** on the Mac run `ipconfig getifaddr en0` (if empty, `ipconfig getifaddr en1`, or System Settings → Wi-Fi → Details → IP address).

## 5. If the Linux laptop cannot reach the Mac

1. **College Wi-Fi often blocks laptop-to-laptop traffic.** Turn on a phone hotspot, connect both laptops to it, get the new IP with `ipconfig getifaddr en0`, and retry section 4.
2. **Mac firewall:** System Settings → Network → Firewall: allow Python, or turn the firewall off for the demo.
3. Check the API window on the Mac is still running and shows no error.
4. The address in the sidebar must start with `http://` and have no space or `/` at the end.

## 6. Starting again later (after a reboot)

```bash
# Mac
cd ~/pathsnap && source .venv/bin/activate && uvicorn api.app:app --host 0.0.0.0 --port 8000
# Linux
cd ~/Downloads/pathsnap_ui && source .venv/bin/activate && streamlit run app.py
```

Check the Mac's IP again; it can change after reconnecting to Wi-Fi.

## 7. Switching between student-test and evaluator trips

The API serves both datasets at once (when their folders exist):

| Dataset | Folder on the Mac |
| --- | --- |
| `student_test` | `data/raw/Student folder/student_test` |
| `evaluator_test` | `data/raw/evaluator_test` (or the folder in `PATHSNAP_EVAL_DIR`) |

- **UI:** choose **Evaluator trips** or **Student-test trips** in the sidebar. Evaluator trips are the default.
- **API log:** every request shows `dataset=... trip=...`.
- **Terminal, whole dataset at once** (saved model and HMM settings; nothing is retrained or saved):

```bash
python run_eval.py --dataset student_test
python run_eval.py --dataset evaluator_test
```

Accuracy is computed only where the `.route` files exist (they stay on the Mac). Without them the route is still predicted and accuracy shows "n/a".
Do not change code, settings or the model after looking at evaluator results.

## 8. During the evaluation (suggested order)

1. **Data (Part 1):** run `python run_data.py`. Walk through steps 2–5 on screen: validation, projection, labels, the 8 candidates of one point, the feature tables. Open `eda/map_train_00000003.html` and `figures/eda_*.png`.
2. **Code:** show `src/geo.py` (projection), `src/labels.py` (order-preserving labels), `src/features.py` (candidates and 9 features).
3. **Models (Part 2):** run `python run_models.py` (about 2.5 min; explain while it runs). Open `experiments/model_comparison.png` and `experiments/confusion_matrix.png`, `figures/optuna_history.png`.
4. **HMM (Part 3):** run `python run_hmm.py`, open `maps/student_test_00000000.html`.
5. **Live system:** evaluator picks a dataset and any trip in the UI on Linux; show the request log on the Mac.
6. **Optional:** `python run_eval.py --dataset evaluator_test` for the table of all 10 evaluator trips.

## 9. Important: every run overwrites the saved results

Each script rewrites its output files (same names, no copies kept):

| Script | Overwrites |
| --- | --- |
| `run_data.py` | `eda/*.csv`, `eda/map_train_00000003.html`, `figures/eda_*.png`, `data/processed/`, `experiments/features_report.json` |
| `run_models.py` | `experiments/model_comparison.*`, `optuna_*`, `tuned_vs_untuned.csv`, `confusion_matrix.png`, `figures/optuna_history.png`, `models/hgb_tuned.joblib` |
| `run_hmm.py` | `experiments/hmm_*.csv`, `models/hmm_params.json`, `maps/student_test_00000000.html` |

- All randomness is seeded (seed 42), so a full re-run gives the **same scores**. Only the time columns (seconds) change slightly.
- Run `run_models.py` **without** `--trials 5` in front of evaluators. A shorter search would overwrite the tuned model and results with a different (weaker) version.
- Once the results are committed, `git status` shows any file a re-run changed, and `git checkout -- experiments models figures maps eda` restores the committed versions.
