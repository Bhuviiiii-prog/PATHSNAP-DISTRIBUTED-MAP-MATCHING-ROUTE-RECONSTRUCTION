"""Shared bits for model comparison and tuning: data loading, metrics, CV."""
import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["dist", "dist_rank", "bearing_diff", "seg_len", "speed", "n_cand_30m"]
SEED = 42
TRAIN_POINT_STRIDE = 4  # GPS is 1 Hz so neighbouring points are near-duplicates; keep every 4th to stay fast


def load(part):
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", f"{part}.parquet"))
    if part == "train":
        df = df[df.point % TRAIN_POINT_STRIDE == 0].reset_index(drop=True)
    return df


def top1_pred(df, score):
    """Candidate-level 0/1 prediction: 1 only for the highest-scoring candidate of each point."""
    s = pd.Series(score, index=df.index)
    best = s.groupby([df.trip, df.point]).transform("max")
    first = (s == best) & ~(s == best).groupby([df.trip, df.point]).cumsum().gt(1)
    return first.astype(int).to_numpy()


def evaluate(model, df):
    t0 = time.time()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    infer = time.time() - t0
    pred = top1_pred(df, proba)
    # Top-1: share of GPS points whose top candidate is the label arc
    # (points whose label arc is not among candidates count as wrong)
    top1 = pd.Series(pred * df.y.to_numpy()).groupby([df.trip, df.point]).sum().mean()
    return {
        "top1_acc": float(top1),
        "macro_f1": float(f1_score(df.y, pred, average="macro")),
        "macro_precision": float(precision_score(df.y, pred, average="macro")),
        "macro_recall": float(recall_score(df.y, pred, average="macro")),
        "log_loss": float(log_loss(df.y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "inference_s": infer,
    }


def cv_macro_f1(make_model, df, folds=3):
    scores = []
    for tr_idx, va_idx in GroupKFold(n_splits=folds).split(df, groups=df.trip):
        m = make_model().fit(df.loc[tr_idx, FEATURES], df.y[tr_idx])
        va = df.loc[va_idx]
        scores.append(f1_score(va.y, top1_pred(va, m.predict_proba(va[FEATURES])[:, 1]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))
