import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.pipeline import FraudDetectionPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas  (IEEE-CIS field names)
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    TransactionID: Optional[int] = Field(None, description="Unique transaction identifier")
    TransactionDT: Optional[int] = Field(
        None, description="Seconds offset from 2017-12-01 (IEEE-CIS reference date)"
    )
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code: W / H / C / S / R")
    card4: Optional[str] = Field(None, description="Card network: visa / mastercard / etc.")
    card6: Optional[str] = Field(None, description="Card type: debit / credit")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    addr1: Optional[float] = Field(None, description="Billing address zip code")
    addr2: Optional[float] = Field(None, description="Billing address country code")
    dist1: Optional[float] = Field(None, description="Distance from home address")
    C1: Optional[float] = Field(None, description="Count of payment methods found (velocity proxy)")
    D1: Optional[float] = Field(None, description="Time delta: days since last transaction")
    DeviceType: Optional[str] = Field(None, description="desktop / mobile / None (missing = fraud signal)")

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionID": 2987000,
                "TransactionDT": 86400,
                "TransactionAmt": 350.00,
                "ProductCD": "C",
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "anonymous.com",
                "R_emaildomain": "gmail.com",
                "addr1": 299.0,
                "addr2": 87.0,
                "dist1": 890.0,
                "C1": 8.0,
                "D1": 1.0,
                "DeviceType": None,
            }
        }


class PredictionResponse(BaseModel):
    TransactionID: Optional[int]
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
        title="IEEE-CIS Fraud Detection API",
        description=(
            "Real-time transaction fraud scoring. "
            "Primary metric: AUC-PR (not accuracy — data is ~3.5% fraud)."
        ),
        version="1.0.0",
    )

    pipeline = FraudDetectionPipeline()
    _model_loaded = False

    @app.on_event("startup")
    async def load_model():
        nonlocal _model_loaded
        try:
            pipeline.trainer.load(model_filename)
            pipeline.preprocessor = DataPreprocessor.load(preprocessor_path)
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
        try:
            result = pipeline.predictor.predict_single(req.model_dump())
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        return PredictionResponse(
            TransactionID=req.TransactionID,
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
                TransactionID=req.transactions[i].TransactionID,
                fraud_probability=r["fraud_probability"],
                is_fraud=r["is_fraud"],
                risk_level=r["risk_level"],
            )
            for i, r in enumerate(raw_results)
        ]
        return BatchResponse(predictions=predictions, count=len(predictions))

    return app


# ---------------------------------------------------------------------------
# Lazy import to avoid circular at startup
# ---------------------------------------------------------------------------
from src.data.preprocessing import DataPreprocessor  # noqa: E402

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
