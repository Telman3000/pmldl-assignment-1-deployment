"""
Stage 3a — Model API (FastAPI)
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_CANDIDATES = [
    Path("/app/models/titanic_model.joblib"),
    Path(__file__).resolve().parent / "models" / "titanic_model.joblib",
]

# Local development fallback: repo_root/models/...
_repo_models = Path(__file__).resolve().parents
if len(_repo_models) > 3:
    MODEL_CANDIDATES.append(_repo_models[3] / "models" / "titanic_model.joblib")

app = FastAPI(title="Titanic Survival API", version="1.0.0")


class PassengerFeatures(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Passenger class")
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0, le=10, description="Siblings/spouses aboard")
    Parch: int = Field(..., ge=0, le=10, description="Parents/children aboard")
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"] = Field(
        ..., description="Port of embarkation: C/Q/S"
    )


class PredictionResponse(BaseModel):
    survived: int
    survival_label: str
    survival_probability: float


def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "Model file not found. Checked: "
        + ", ".join(str(p) for p in MODEL_CANDIDATES)
        + ". Run model engineering first."
    )


model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PassengerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0][1])
    survived = int(model.predict(row)[0])

    return PredictionResponse(
        survived=survived,
        survival_label="Survived" if survived == 1 else "Did not survive",
        survival_probability=round(proba, 4),
    )
