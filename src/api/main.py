"""
main.py — FastAPI serving layer for the heart disease classifier.

Endpoints:
  GET  /health   liveness/readiness probe target for Docker + Kubernetes
  POST /predict  accepts patient features as JSON, returns prediction + confidence
  GET  /metrics  Prometheus scrape target (added by the instrumentator below)

Run locally: uvicorn src.api.main:app --reload --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")

ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

_pipeline = None
_metadata = {}


def _load_model() -> None:
    global _pipeline, _metadata
    if not MODEL_PATH.exists():
        logger.error(f"model file not found at {MODEL_PATH} — did you run src/train.py?")
        return
    _pipeline = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        _metadata = json.loads(METADATA_PATH.read_text())
    logger.info(f"loaded model={_metadata.get('best_model', 'unknown')} "
                f"test_roc_auc={_metadata.get('metrics', {}).get('roc_auc', 'n/a')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="MLOps AIMLCZG523 Assignment 01 — predicts presence of heart disease from patient features.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)  # adds GET /metrics


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels colored by flouroscopy (0-3)")
    thal: float = Field(..., description="Thalassemia: 3=normal, 6=fixed defect, 7=reversible defect")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
        }
    })


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    model_used: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _pipeline is not None else "model_not_loaded"}


@app.get("/")
def root() -> dict:
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    row = pd.DataFrame([features.model_dump()])
    try:
        pred = int(_pipeline.predict(row)[0])
        proba = float(_pipeline.predict_proba(row)[0][1])
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc

    logger.info(f"predict request={features.model_dump()} -> prediction={pred} probability={proba:.4f}")

    return PredictionResponse(
        prediction=pred,
        label="Disease" if pred == 1 else "No disease",
        probability=round(proba, 4),
        model_used=_metadata.get("best_model", "unknown"),
    )
