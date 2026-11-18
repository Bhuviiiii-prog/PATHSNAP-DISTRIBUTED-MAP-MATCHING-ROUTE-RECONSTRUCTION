"""Train 5 models, 3-fold GroupKFold CV on train, evaluate on student_test.
Output: experiments/model_comparison.csv, experiments/model_comparison.png
Run on its own: .venv/bin/python modelling/train_compare.py   (run_models.py calls these functions)
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
from common import ROOT, FEATURES, SEED, TRAIN_POINT_STRIDE, load, evaluate, cv_macro_f1

# scaler sits inside the pipeline, so it is fit on training folds only
MODELS = {
    "LogisticRegression": ("linear", "C=1.0, max_iter=500, scaled",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
    "GaussianNB": ("probabilistic", "defaults, scaled",
                   lambda: make_pipeline(StandardScaler(), GaussianNB())),
    "KNN": ("instance-based", "k=25, distance-weighted, scaled",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
    "RandomForest": ("bagging ensemble", "100 trees, min_samples_leaf=5",
                     lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": ("boosting", "sklearn defaults",
                             lambda: HistGradientBoostingClassifier(random_state=SEED)),
}


def run_one(name, train, test):
    family, config, make = MODELS[name]
    cv_mean, cv_std = cv_macro_f1(make, train)
    t0 = time.time()
    model = make().fit(train[FEATURES], train.y)
    train_s = time.time() - t0
    return {"model": name, "family": family, "config": config,
            "cv_macro_f1": cv_mean, "cv_macro_f1_std": cv_std, "train_s": train_s, **evaluate(model, test)}


def save(rows):
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "experiments")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "model_comparison.csv"), index=False)
    json.dump({"seed": SEED, "features": FEATURES, "cv": "GroupKFold(3) by trip", "train_point_stride": TRAIN_POINT_STRIDE},
              open(os.path.join(out, "model_comparison_config.json"), "w"), indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    df.plot.bar(x="model", y=["cv_macro_f1", "macro_f1", "top1_acc"], ax=axes[0], ylim=(0.5, 1), rot=30,
                title="CV macro F1 (train) / test macro F1 / test Top-1")
    df.plot.bar(x="model", y="log_loss", ax=axes[1], rot=30, title="Student-test log loss (lower is better)", legend=False)
    df.plot.bar(x="model", y="inference_s", ax=axes[2], rot=30, title="Inference time, s (lower is better)", legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "model_comparison.png"), dpi=120)
    plt.close()
    return df


if __name__ == "__main__":
    train, test = load("train"), load("student_test")
    print("train rows", len(train), "test rows", len(test))
    rows = []
    for name in MODELS:
        res = run_one(name, train, test)
        rows.append(res)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "config"})
    save(rows)
    print("saved experiments/model_comparison.csv and .png")
