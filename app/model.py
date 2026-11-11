import os
from functools import lru_cache

import joblib
import pandas as pd


def model_path() -> str:
    return os.getenv("MODEL_PATH", "models/model.joblib")


def model_available() -> bool:
    return os.path.exists(model_path())


@lru_cache(maxsize=2)
def _load(path: str):
    return joblib.load(path)


def artifact() -> dict:
    return _load(model_path())


def predict_proba(tx: dict) -> float:
    art = artifact()
    row = pd.DataFrame([tx])[art["features"]]
    return float(art["model"].predict_proba(row)[0, 1])


def risk_label(p: float, threshold: float) -> str:
    if p >= threshold:
        return "high"
    return "medium" if p >= threshold / 2 else "low"
