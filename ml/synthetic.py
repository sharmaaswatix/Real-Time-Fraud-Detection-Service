import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_synthetic(n: int = 6000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[:20] = 1
    X = rng.normal(size=(n, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 13, 16]] = [-3.0, -3.5, 2.5, -3.0, -4.0, -3.0]
    X[y == 1] += shift
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = np.round(rng.lognormal(3.0, 1.2, n), 2)
    df["Class"] = y
    return df
