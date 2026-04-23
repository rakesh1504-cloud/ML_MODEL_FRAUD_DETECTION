# ML Model — Fraud Detection Pipeline

End-to-end machine learning pipeline for real-time credit card fraud detection. Covers data ingestion, feature engineering, model training, evaluation, and REST API serving.

---

## Project Structure

```
ML_MODEL_FRAUD_DETECTION/
├── data/
│   ├── raw/                    # Raw transaction data (CSV / Parquet)
│   ├── processed/              # Fitted preprocessor (pickle)
│   └── models/                 # Trained model artifacts + evaluation report
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Load files or generate synthetic data
│   │   └── preprocessing.py    # Clean, encode, scale, train/val/test split
│   ├── features/
│   │   └── build_features.py   # Time, amount, risk, velocity features
│   ├── models/
│   │   ├── train.py            # ModelTrainer (LR / RF / GBM)
│   │   ├── evaluate.py         # Metrics, threshold optimisation
│   │   └── predict.py          # FraudPredictor (single + batch)
│   └── pipeline.py             # FraudDetectionPipeline orchestrator
├── api/
│   └── app.py                  # FastAPI app — /predict, /predict/batch, /health
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_pipeline.py
├── config/
│   └── config.yaml             # All knobs in one place
├── notebooks/                  # Jupyter notebooks for EDA
├── requirements.txt
├── setup.py
└── Makefile
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
# or
make install
```

### 2. Train the model

```bash
# Uses synthetic data (no CSV needed)
make train

# Or with your own CSV file
make train-file FILE=data/raw/transactions.csv
```

### 3. Run the API

```bash
make serve
# Swagger UI → http://localhost:8000/docs
```

### 4. Run tests

```bash
make test
```

---

## Usage — Python API

```python
from src.pipeline import FraudDetectionPipeline

pipeline = FraudDetectionPipeline()

# Train on synthetic data
summary = pipeline.run_training(model_name="random_forest")
print(summary["test_metrics"])

# Score a single transaction
result = pipeline.predict_single({
    "amount": 1500.0,
    "hour": 3,
    "day_of_week": 6,
    "merchant_category": "online",
    "card_present": 0,
    "distance_from_home_km": 900.0,
    "num_transactions_last_24h": 12,
    "is_foreign_transaction": 1,
})
print(result)
# {'fraud_probability': 0.87, 'is_fraud': True, 'risk_level': 'HIGH'}
```

---

## REST API Endpoints

| Method | Endpoint          | Description              |
|--------|-------------------|--------------------------|
| GET    | `/health`         | Liveness + model status  |
| POST   | `/predict`        | Score a single transaction |
| POST   | `/predict/batch`  | Score multiple transactions |

Full schema available at `/docs` after starting the server.

---

## Supported Models

| Key                   | Algorithm                  |
|-----------------------|----------------------------|
| `logistic_regression` | Logistic Regression        |
| `random_forest`       | Random Forest (default)    |
| `gradient_boosting`   | Gradient Boosting (sklearn)|

Configure via `config/config.yaml` → `model.name`.

---

## Key Features

- **Imbalanced-class handling** — `class_weight="balanced"` on all classifiers
- **Threshold optimisation** — sweeps F1 / recall / precision on validation set
- **Feature engineering** — cyclic time encoding, log-amount, geo-risk, velocity flags
- **Reproducible splits** — stratified train / val / test with fixed seeds
- **Pluggable models** — swap algorithms via config with zero code change
- **Production-ready API** — FastAPI with Pydantic validation and batch endpoint
