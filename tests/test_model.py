import numpy as np

from app import model
from ml.synthetic import FEATURES, make_synthetic
from ml.train import pick_threshold


def test_training_reports_expected_metrics(env):
    best = env["models"][env["best_model"]]["test"]
    assert {"recall", "precision", "pr_auc", "roc_auc", "threshold"} <= best.keys()
    assert best["recall"] > 0.5 and best["pr_auc"] > 0.5


def test_model_output_contract(env):
    rows = make_synthetic(n=300, seed=7)[FEATURES].to_dict("records")
    probs = [model.predict_proba(r) for r in rows]
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_risk_label_bands():
    assert model.risk_label(0.9, 0.5) == "high"
    assert model.risk_label(0.3, 0.5) == "medium"
    assert model.risk_label(0.1, 0.5) == "low"


def test_recall_weighting_never_raises_threshold():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    p = np.clip(np.concatenate([rng.normal(0.25, 0.15, 900), rng.normal(0.65, 0.2, 100)]), 0, 1)
    assert pick_threshold(y, p, beta=2.0) <= pick_threshold(y, p, beta=1.0)
