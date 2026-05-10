import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root — where uvicorn is launched from)
# ---------------------------------------------------------------------------
MODEL_PATH     = "models/lgbm_champion.pkl"
THRESHOLD_PATH = "models/threshold.json"
FEATURES_PATH  = "data/processed/feature_cols.json"

# ---------------------------------------------------------------------------
# Global state loaded once at startup
# ---------------------------------------------------------------------------
model        = None
threshold    = 0.5
feature_cols: List[str] = []
explainer    = None
model_meta: Dict = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time transaction fraud scoring — LightGBM + SHAP explanations.\n\n"
        "Primary metric: **AUC-PR** (accuracy is meaningless at 3.5% fraud rate).\n"
        "Dataset: IEEE-CIS Fraud Detection (Kaggle, 590k transactions)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    TransactionAmt: float          = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD:      Optional[str]  = Field(None, description="Product code: W/H/C/S/R")
    card1:          Optional[int]  = Field(None, description="Card identifier (anonymised)")
    card4:          Optional[str]  = Field(None, description="Card network: visa/mastercard/discover/american express")
    card6:          Optional[str]  = Field(None, description="Card type: debit/credit")
    P_emaildomain:  Optional[str]  = Field(None, description="Purchaser email domain")
    R_emaildomain:  Optional[str]  = Field(None, description="Recipient email domain")
    addr1:          Optional[float] = Field(None, description="Billing zip code")
    C1:             Optional[float] = Field(None, description="Count of payment methods (velocity proxy)")
    D1:             Optional[float] = Field(None, description="Days since last transaction")
    DeviceType:     Optional[str]  = Field(None, description="desktop/mobile — missing is a fraud signal")
    TransactionID:  Optional[int]  = Field(None, description="Unique transaction identifier")
    TransactionDT:  Optional[int]  = Field(None, description="Seconds offset from 2017-12-01")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 350.0,
                "ProductCD": "C",
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "anonymous.com",
                "C1": 8.0,
                "DeviceType": None,
                "TransactionID": 2987000,
            }
        }
    }


class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    risk_level:        str
    top_shap_features: Dict[str, float]
    model_version:     str
    TransactionID:     Optional[int] = None


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPrediction(BaseModel):
    predictions: List[FraudPrediction]
    count:       int


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    threshold:     float
    n_features:    int
    model_version: str


class ModelInfoResponse(BaseModel):
    model_type:  str
    threshold:   float
    auc_pr:      Optional[float]
    recall:      Optional[float]
    n_features:  int
    dataset:     str
    version:     str


# ---------------------------------------------------------------------------
# Startup: load model, threshold, features, SHAP explainer
# ---------------------------------------------------------------------------

def _load_model():
    global model, threshold, feature_cols, explainer, model_meta

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Champion model not found at {MODEL_PATH} — API will return 503 until loaded.")
        return

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded: {type(model).__name__}")

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            model_meta = json.load(f)
        threshold = model_meta.get("threshold", 0.5)
        logger.info(f"Threshold loaded: {threshold:.4f}")

    # Prefer feature names embedded in the model (LightGBM stores them after fit).
    # Fall back to the JSON file only if the model doesn't expose them.
    if hasattr(model, "feature_name_") and model.feature_name_:
        feature_cols = list(model.feature_name_)
        logger.info(f"Feature columns from model: {len(feature_cols)}")
    elif os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            feature_cols = json.load(f)
        logger.info(f"Feature columns from {FEATURES_PATH}: {len(feature_cols)}")

    # SHAP TreeExplainer — lazy import so shap is optional at import time
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialised")
    except Exception as exc:
        logger.warning(f"SHAP explainer not available: {exc}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _risk_level(prob: float, is_fraud: bool) -> str:
    if prob >= 0.80:
        return "CRITICAL"
    if is_fraud:
        return "HIGH"
    if prob >= 0.20:
        return "MEDIUM"
    if prob >= 0.05:
        return "LOW"
    return "VERY_LOW"


def _build_row(request: TransactionRequest) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_cols}
    for field, value in request.model_dump().items():
        if field in row and value is not None:
            try:
                row[field] = float(value)
            except (ValueError, TypeError):
                pass  # non-numeric request field — leave numeric default 0.0
    return pd.DataFrame([row])


def _top_shap(df: pd.DataFrame, n: int = 5) -> Dict[str, float]:
    if explainer is None:
        return {}
    try:
        sv = explainer.shap_values(df)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]
        top_idx = np.argsort(np.abs(sv))[-n:][::-1]
        return {feature_cols[i]: round(float(sv[i]), 4) for i in top_idx}
    except Exception as exc:
        logger.warning(f"SHAP failed: {exc}")
        return {}


def _score(request: TransactionRequest) -> FraudPrediction:
    df       = _build_row(request)
    prob     = float(model.predict_proba(df)[0, 1])
    is_fraud = prob >= threshold
    return FraudPrediction(
        fraud_probability=round(prob, 4),
        is_fraud=bool(is_fraud),
        threshold_used=round(threshold, 4),
        risk_level=_risk_level(prob, is_fraud),
        top_shap_features=_top_shap(df),
        model_version=model_meta.get("model_version", "lgbm-v1.0"),
        TransactionID=request.TransactionID,
    )


def _check_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train and save a champion model first.",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        threshold=threshold,
        n_features=len(feature_cols),
        model_version=model_meta.get("model_version", "lgbm-v1.0"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info():
    _check_model_loaded()
    return ModelInfoResponse(
        model_type=type(model).__name__,
        threshold=threshold,
        auc_pr=model_meta.get("auc_pr"),
        recall=model_meta.get("recall_at_threshold"),
        n_features=len(feature_cols),
        dataset="IEEE-CIS Fraud Detection (Kaggle)",
        version="1.0.0",
    )


@app.post("/predict", response_model=FraudPrediction, tags=["inference"])
def predict(request: TransactionRequest):
    _check_model_loaded()
    try:
        return _score(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(req: BatchRequest):
    _check_model_loaded()
    try:
        predictions = [_score(txn) for txn in req.transactions]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return BatchPrediction(predictions=predictions, count=len(predictions))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
