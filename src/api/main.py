

import json
import os
import joblib

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["MATPLOTLIBRC"] = "/tmp"

# Prevent matplotlib font cache from building

matplotlib.use("Agg")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring -- LightGBM",
    version="1.0.0"
)

MODEL_PATH     = "models/lgbm_champion.pkl"
THRESHOLD_PATH = "models/threshold.json"
FEATURES_PATH  = "data/processed/feature_cols.json"

model        = None
threshold    = 0.5
feature_cols = []


class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0)
    ProductCD:      Optional[str]   = None
    card1:          Optional[int]   = None
    card4:          Optional[str]   = None
    card6:          Optional[str]   = None
    P_emaildomain:  Optional[str]   = None
    addr1:          Optional[float] = None


class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    risk_level:        str
    model_version:     str


@app.on_event("startup")
def load_model():
    global model, threshold, feature_cols
    # Load ONLY model and threshold -- no SHAP at startup
    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    print(f"Model loaded | threshold={threshold:.3f}")


@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "threshold"   : threshold,
        "n_features"  : len(feature_cols)
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type"    : "LightGBMClassifier",
        "threshold"     : threshold,
        "n_features"    : len(feature_cols),
        "version"       : "1.0.0",
        "dataset"       : "IEEE-CIS Fraud Detection",
        "primary_metric": "AUC-PR 0.86"
    }


@app.post("/predict", response_model=FraudPrediction)
def predict(request: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build input row
    row = {col: 0 for col in feature_cols}
    for k, v in request.model_dump().items():
        if k in row and v is not None:
            row[k] = v
    df = pd.DataFrame([row])

    # Score only -- no SHAP (saves ~150MB RAM)
    prob     = float(model.predict_proba(df)[0, 1])
    is_fraud = prob >= threshold
    risk     = (
        "CRITICAL" if prob >= 0.80 else
        "HIGH"     if is_fraud     else
        "MEDIUM"   if prob >= 0.20 else
        "LOW"
    )

    return FraudPrediction(
        fraud_probability = round(prob, 4),
        is_fraud          = bool(is_fraud),
        threshold_used    = round(threshold, 3),
        risk_level        = risk,
        model_version     = "lgbm-v1.0"
    )


if __name__ == "__main__":
    uvicorn.run("src.api.main:app",
                host="0.0.0.0", port=8000, reload=True)
