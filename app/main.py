import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from app import db, model
from app.auth import check_credentials, create_token, current_user
from app.schemas import PredictionOut, StoredPrediction, Token, Transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fraud-api")

app = FastAPI(title="Fraud Detection Service", version="1.0.0",
              description="Scores card transactions with an ML model and keeps an audit log.")
app.add_middleware(origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"]))


def _require_model():
    if not model.model_available():
        raise HTTPException(503, "Model not trained yet. Run: python -m ml.train")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model.model_available()}


@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    if not check_credentials(form.username, form.password):
        raise HTTPException(401, "Incorrect username or password")
    return Token(access_token=create_token(form.username))


@app.get("/model/info")
def model_info():
    _require_model()
    return model.artifact()["metrics"]


@app.get("/samples")
def samples():
    _require_model()
    return model.artifact()["samples"]


@app.post("/predict", response_model=PredictionOut)
def predict(tx: Transaction, user: str = Depends(current_user)):
    _require_model()
    data = tx.model_dump()
    p = model.predict_proba(data)
    thr = model.artifact()["threshold"]
    flagged, label = p >= thr, model.risk_label(p, thr)
    pid = db.insert_prediction(user, data, p, thr, flagged, label)
    log.info("prediction id=%s user=%s p=%.4f flagged=%s", pid, user, p, flagged)
    return PredictionOut(id=pid, fraud_probability=p, threshold=thr, flagged=flagged, risk_label=label)


@app.get("/transactions", response_model=list[StoredPrediction])
def transactions(limit: int = Query(50, ge=1, le=500), flagged_only: bool = False,
                 user: str = Depends(current_user)):
    return db.list_predictions(limit, flagged_only)
