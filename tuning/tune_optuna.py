"""Optuna TPE tuning of HistGradientBoosting (best CV macro F1 in model_comparison.csv).
Objective: mean 3-fold GroupKFold (by trip) macro F1 on train only.
Output: experiments/optuna_trials.csv, optuna_summary.json, tuned_vs_untuned.csv,
        confusion_matrix.png, models/hgb_tuned.joblib
Run on its own: .venv/bin/python tuning/tune_optuna.py   (run_models.py calls these functions)
"""
import os, sys, time, json
import joblib
import optuna
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelling"))
from common import ROOT, FEATURES, SEED, load, evaluate, cv_macro_f1, top1_pred

N_TRIALS = 15

SEARCH_SPACE = {
    "learning_rate": "log-uniform [0.02, 0.3]",
    "max_iter": "int [100, 400]",
    "max_leaf_nodes": "int [8, 64]",
    "min_samples_leaf": "int [20, 300]",
    "l2_regularization": "log-uniform [1e-4, 10]",
}


def make(params):
    return lambda: HistGradientBoostingClassifier(random_state=SEED, **params)


def objective(trial, train):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 400),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
    }
    return cv_macro_f1(make(params), train)[0]


def run_search(train, n_trials=N_TRIALS, on_trial=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    callbacks = [lambda st, tr: on_trial(st, tr)] if on_trial else None
    study.optimize(lambda tr: objective(tr, train), n_trials=n_trials, callbacks=callbacks)
    search_s = time.time() - t0
    out = os.path.join(ROOT, "experiments")
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
    json.dump({"search_space": SEARCH_SPACE, "objective": "mean macro F1, GroupKFold(3) by trip, train only",
               "sampler": "TPE", "seed": SEED, "n_trials": n_trials, "search_seconds": search_s,
               "best_params": study.best_params, "best_cv_macro_f1": study.best_value},
              open(os.path.join(out, "optuna_summary.json"), "w"), indent=2)
    return study, search_s


def compare(train, test, best_params):
    rows, tuned_model = [], None
    for name, params in [("untuned", {}), ("tuned", best_params)]:
        cv_mean, _ = cv_macro_f1(make(params), train)
        t1 = time.time()
        model = make(params)().fit(train[FEATURES], train.y)
        rows.append({"model": f"HGB_{name}", "cv_macro_f1": cv_mean, "train_s": time.time() - t1, **evaluate(model, test)})
        if name == "tuned":
            tuned_model = model
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "experiments", "tuned_vs_untuned.csv"), index=False)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(tuned_model, os.path.join(ROOT, "models", "hgb_tuned.joblib"))
    return df, tuned_model


def confusion(model, test):
    """Candidate-level confusion matrix of the tuned model on student_test (top-1 decision rule)."""
    pred = top1_pred(test, model.predict_proba(test[FEATURES])[:, 1])
    cm = confusion_matrix(test.y, pred)
    ConfusionMatrixDisplay(cm, display_labels=["wrong arc (0)", "right arc (1)"]).plot(values_format="d", cmap="Blues")
    plt.title("Tuned HGB, student_test candidate rows")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "experiments", "confusion_matrix.png"), dpi=120)
    plt.close()
    return cm


if __name__ == "__main__":
    train, test = load("train"), load("student_test")
    study, _ = run_search(train)
    df, model = compare(train, test, study.best_params)
    confusion(model, test)
    print("best params", study.best_params)
    print(df.round(4).to_string(index=False))
