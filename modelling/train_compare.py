"""Train 5 models, 3-fold GroupKFold CV on train, evaluate on student_test.
Output: experiments/model_comparison.csv, experiments/model_comparison.png
Run: .venv/bin/python modelling/train_compare.py
"""
import os, sys, time, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, FEATURES, SEED, load, evaluate, cv_macro_f1

# scaler sits inside the pipeline, so it is fit on training folds only
MODELS = {
    "LogisticRegression": ("linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
    "GaussianNB": ("probabilistic", lambda: make_pipeline(StandardScaler(), GaussianNB())),
    "KNN": ("instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
    "RandomForest": ("bagging ensemble", lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": ("boosting", lambda: HistGradientBoostingClassifier(random_state=SEED)),
}

if __name__ == "__main__":
    train, test = load("train"), load("student_test")
    print("train rows", len(train), "test rows", len(test))
    rows = []
    for name, (family, make) in MODELS.items():
        cv_mean, cv_std = cv_macro_f1(make, train)
        t0 = time.time()
        model = make().fit(train[FEATURES], train.y)
        train_s = time.time() - t0
        res = {"model": name, "family": family, "config": str(model.get_params() if name != "LogisticRegression" else "C=1.0, max_iter=500")[:120],
               "cv_macro_f1": cv_mean, "cv_macro_f1_std": cv_std, "train_s": train_s, **evaluate(model, test)}
        rows.append(res)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "config"})
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "experiments")
    df.to_csv(os.path.join(out, "model_comparison.csv"), index=False)
    json.dump({"seed": SEED, "features": FEATURES, "cv": "GroupKFold(3) by trip", "train_point_stride": 4}, open(os.path.join(out, "model_comparison_config.json"), "w"), indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.plot.bar(x="model", y=["cv_macro_f1", "macro_f1", "top1_acc"], ax=axes[0], ylim=(0.5, 1), rot=30,
                title="CV macro F1 / test macro F1 / test Top-1")
    df.plot.bar(x="model", y="log_loss", ax=axes[1], rot=30, title="Student-test log loss", legend=False)
    df.plot.bar(x="model", y="inference_s", ax=axes[2], rot=30, title="Inference time (s, 153k rows)", legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "model_comparison.png"), dpi=120)
    print("saved experiments/model_comparison.csv and .png")
