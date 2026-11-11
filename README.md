# Real-Time Fraud Detection Service

An ML fraud classifier served through a FastAPI REST API, with JWT auth, a SQL audit log, a React review dashboard, pytest tests and Docker.

**Stack:** Python · FastAPI · XGBoost · scikit-learn · SQLite · React (Vite) · pytest · Docker

## Why this project is built the way it is

- **Imbalanced data.** The Kaggle dataset has 492 frauds in 284,807 transactions (0.17%). Accuracy is meaningless here (predicting "legit" always scores 99.8%), so models are compared on **PR-AUC** and judged on **recall and precision**.
- **Class imbalance handling.** Logistic regression uses `class_weight="balanced"`; XGBoost uses `scale_pos_weight`.
- **Threshold tuning.** The decision threshold is picked on a validation set by maximising F2 (recall counts double), because a missed fraud costs more than a manual review. It is never tuned on the test set.
- **Audit log.** Every prediction (input, score, threshold, decision, user, time) is stored in SQLite.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download creditcard.csv from
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  and put it in data/
python -m ml.train --csv data/creditcard.csv      # prints metrics, saves models/model.joblib

# 2. Run the API (docs at http://localhost:8000/docs)
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DEMO_PASSWORD="choose-a-password"
uvicorn app.main:app --reload

# 3. Run the dashboard (new terminal)
cd frontend && npm install && npm run dev          # http://localhost:5173
```

Sign in as `analyst` with your `DEMO_PASSWORD`.

**Docker:** train the model first, copy `.env.example` to `.env`, then `docker compose up --build`. API on :8000, dashboard on :8080.

**Tests:** `pytest` (uses a small synthetic dataset, so no download needed).

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness and whether a model is loaded |
| POST | `/auth/login` | no | Exchange username/password for a JWT |
| POST | `/predict` | JWT | Score one transaction, log it |
| GET | `/transactions` | JWT | Recent predictions (`flagged_only`, `limit`) |
| GET | `/samples`, `/model/info` | no | Sample transactions and training metrics |

Input is validated with Pydantic: all 30 fields required, `Amount` and `Time` ≥ 0, unknown fields rejected.

## Results

Fill these in from the output of `python -m ml.train` on the real dataset (test split):

| Model | Recall | Precision | PR-AUC |
|---|---|---|---|
| Logistic regression (baseline) | | | |
| XGBoost | | | |

## Limitations

- The dataset's V1–V28 features are anonymised PCA components, so predictions can't be explained in business terms.
- Auth is a single demo account from environment variables. A real system would use a users table with hashed passwords.
- SQLite is fine for a demo; a production service would use a server database and run multiple workers.
