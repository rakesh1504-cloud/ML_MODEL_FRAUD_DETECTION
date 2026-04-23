import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.pipeline import FraudDetectionPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    merchant_category: str = Field(..., description="Merchant category")
    card_present: int = Field(..., ge=0, le=1, description="1 if physical card was used")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from cardholder home")
    num_transactions_last_24h: int = Field(..., ge=0, description="Velocity: txns in last 24h")
    is_foreign_transaction: int = Field(..., ge=0, le=1, description="1 if foreign transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN00001234",
                "amount": 350.00,
                "hour": 2,
                "day_of_week": 6,
                "merchant_category": "online",
                "card_present": 0,
                "distance_from_home_km": 850.0,
                "num_transactions_last_24h": 8,
                "is_foreign_transaction": 1,
            }
        }


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_filename: str = "random_forest.pkl",
    preprocessor_path: str = "data/processed/preprocessor.pkl",
    threshold: float = 0.5,
) -> FastAPI:
    app = FastAPI(
        title="Fraud Detection API",
        description="Real-time transaction fraud scoring powered by ML.",
        version="1.0.0",
    )

    pipeline = FraudDetectionPipeline()
    _model_loaded = False

    @app.on_event("startup")
    async def load_model():
        nonlocal _model_loaded
        try:
            pipeline.trainer.load(model_filename)
            pipeline.preprocessor = pipeline.preprocessor.load(preprocessor_path)
            from src.models.predict import FraudPredictor
            pipeline.predictor = FraudPredictor.from_components(
                pipeline.trainer, pipeline.preprocessor, threshold
            )
            _model_loaded = True
            logger.info("Model loaded successfully at startup")
        except Exception as exc:
            logger.warning(f"Model not loaded at startup: {exc}")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        return {"status": "ok", "model_loaded": _model_loaded}

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(req: TransactionRequest):
        if not _model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Run the training pipeline first.",
            )
        transaction = req.model_dump()
        try:
            result = pipeline.predictor.predict_single(transaction)
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        return PredictionResponse(
            transaction_id=req.transaction_id,
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
        )

    @app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
    async def predict_batch(req: BatchRequest):
        if not _model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Run the training pipeline first.",
            )
        import pandas as pd
        df = pd.DataFrame([t.model_dump() for t in req.transactions])
        try:
            raw_results = pipeline.predictor.predict_batch(df)
        except Exception as exc:
            logger.error(f"Batch prediction error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        predictions = [
            PredictionResponse(
                transaction_id=req.transactions[i].transaction_id,
                fraud_probability=r["fraud_probability"],
                is_fraud=r["is_fraud"],
                risk_level=r["risk_level"],
            )
            for i, r in enumerate(raw_results)
        ]
        return BatchResponse(predictions=predictions, count=len(predictions))

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
