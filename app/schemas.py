from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

_fields = {"Time": (float, Field(ge=0, description="Seconds since first transaction"))}
_fields.update({f"V{i}": (float, Field(description=f"Anonymised PCA feature {i}")) for i in range(1, 29)})
_fields["Amount"] = (float, Field(ge=0, description="Transaction amount"))

Transaction = create_model("Transaction", __config__=ConfigDict(extra="forbid"), **_fields)


class PredictionOut(BaseModel):
    id: int
    fraud_probability: float
    threshold: float
    flagged: bool
    risk_label: Literal["low", "medium", "high"]


class StoredPrediction(PredictionOut):
    created_at: datetime
    username: str
    amount: float


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
