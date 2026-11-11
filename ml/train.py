import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.synthetic import FEATURES, make_synthetic


def pick_threshold(y_true, proba, beta: float = 2.0) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    p, r = p[:-1], r[:-1]
    f = (1 + beta**2) * p * r / np.maximum(beta**2 * p + r, 1e-12)
    return float(t[int(np.argmax(f))])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
    }


def train(csv_path=None, synthetic=False, out_dir="models", seed=42, n_estimators=300) -> dict:
    df = make_synthetic(seed=seed) if synthetic else pd.read_csv(csv_path)
    X, y = df[FEATURES], df["Class"].to_numpy()

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "xgboost": XGBClassifier(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.1,
            scale_pos_weight=neg / max(pos, 1), eval_metric="aucpr",
            tree_method="hist", random_state=seed, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        thr = pick_threshold(y_val, val_proba)
        results[name] = {
            "threshold": thr,
            "validation": evaluate(y_val, val_proba, thr),
            "test": evaluate(y_test, model.predict_proba(X_test)[:, 1], thr),
        }

    best = max(results, key=lambda n: results[n]["validation"]["pr_auc"])
    test_df = X_test.assign(Class=y_test)
    samples = {
        "fraud": test_df[test_df.Class == 1].head(3)[FEATURES].to_dict("records"),
        "legit": test_df[test_df.Class == 0].head(3)[FEATURES].to_dict("records"),
    }
    summary = {"best_model": best, "n_train": len(X_train), "fraud_rate": float(y.mean()), "models": results}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": models[best], "threshold": results[best]["threshold"],
                 "features": FEATURES, "samples": samples, "metrics": summary}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    s = train(None if a.synthetic else a.csv, a.synthetic, a.out)
    for name, r in s["models"].items():
        t = r["test"]
        print(f"{name:20s} test recall={t['recall']:.3f} precision={t['precision']:.3f} PR-AUC={t['pr_auc']:.3f}")
    print("saved best model:", s["best_model"])
