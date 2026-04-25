# ML Model — Fraud Detection Pipeline

An end-to-end machine learning pipeline for real-time credit card fraud detection, built on the **IEEE-CIS Fraud Detection** dataset (Kaggle). Covers Kaggle data ingestion, exploratory analysis, feature engineering, model training, evaluation, and REST API serving.

> **Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — 590k transactions, 433 features, ~3.5% fraud rate.

---

## Project Structure

```
ML_MODEL_FRAUD_DETECTION/
├── data/
│   ├── raw/                         # IEEE-CIS CSV files from Kaggle
│   ├── processed/                   # Parquet splits + preprocessor pickle
│   ├── models/                      # Trained model artifacts + evaluation report
│   └── external/                    # EDA & feature importance charts (notebooks)
├── src/
│   ├── data/
│   │   ├── ingestion.py             # Load & merge transaction + identity files
│   │   └── preprocessing.py        # Drop/flag missing, impute, time-split, SMOTE
│   ├── features/
│   │   └── build_features.py       # Velocity, amount aggs, interactions, encoding
│   ├── models/
│   │   ├── train.py                 # LR / RF / GBM / LightGBM trainer
│   │   ├── evaluate.py              # AUC-PR, ROC-AUC, F1, threshold optimisation
│   │   └── predict.py              # FraudPredictor — single & batch inference
│   └── pipeline.py                  # FraudDetectionPipeline orchestrator
├── api/
│   └── app.py                       # FastAPI — /predict, /predict/batch, /health
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA — class imbalance, missing values, patterns
│   └── 01_feature_engineering.ipynb # Feature engineering walkthrough
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_pipeline.py
├── config/
│   └── config.yaml                  # All knobs in one place
├── requirements.txt
├── setup.py
└── Makefile
```

---

## Getting the Data from Kaggle

### Step 1 — Set up Kaggle API credentials

1. Sign in to [kaggle.com](https://www.kaggle.com) and go to **Account → API → Create New Token**
2. This downloads `kaggle.json` to your machine
3. Place it in the correct location:

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

### Step 2 — Install the Kaggle CLI

```bash
pip install kaggle
```

### Step 3 — Accept the competition rules

Visit the competition page, scroll to the bottom, and click **"I Understand and Accept"**.
You must do this once before the API will allow downloads.

### Step 4 — Download and unzip the dataset

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip
```

After unzipping you should have:

```
data/raw/
├── train_transaction.csv   # 590,540 rows × 394 columns
├── train_identity.csv      # 144,233 rows × 41 columns
├── test_transaction.csv    # for inference / submission
└── test_identity.csv
```

> **Tip:** The zip is ~360 MB. The merged DataFrame is ~2 GB in memory — 16 GB RAM recommended.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
# or
make install
```

### 2. Download data (see above) then train

```bash
# Train on real IEEE-CIS data
python -c "
from src.pipeline import FraudDetectionPipeline
FraudDetectionPipeline().run_training(
    txn_file='train_transaction.csv',
    identity_file='train_identity.csv',
    model_name='lightgbm',
)
"

# Or use synthetic data (no Kaggle account needed)
make train
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

# Train on IEEE-CIS data
summary = pipeline.run_training(
    txn_file="train_transaction.csv",
    identity_file="train_identity.csv",
    model_name="lightgbm",
    apply_smote=True,
)
print(summary["test_metrics"])
# {'auc_pr': 0.82, 'roc_auc': 0.95, 'f1': 0.71, ...}

# Score a single transaction
result = pipeline.predict_single({
    "TransactionAmt": 1500.0,
    "TransactionDT": 86400,
    "ProductCD": "C",
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "gmail.com",
    "C1": 12.0,
    "DeviceType": None,
})
print(result)
# {'fraud_probability': 0.87, 'is_fraud': True, 'risk_level': 'HIGH'}
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + model loaded status |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score multiple transactions |

Full schema and examples available at `/docs` after starting the server.

---

## Supported Models

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `lightgbm` | LightGBM | Default — fastest, best AUC-PR |
| `random_forest` | Random Forest | Good baseline, interpretable |
| `gradient_boosting` | sklearn GBM | Slower, useful for comparison |
| `logistic_regression` | Logistic Regression | Interpretable, fast to train |

Configure via `config/config.yaml` → `model.name`.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **AUC-PR as primary metric** | Accuracy is misleading at 3.5% fraud rate — a model predicting "never fraud" scores 96.5% |
| **Time-based train/test split** | Fraud patterns evolve; random splits leak future data into training |
| **SMOTE after split, on train only** | Applying before split leaks synthetic fraud samples into the test set |
| **CV target encoding** | Prevents label leakage when encoding high-cardinality email domains |
| **Missingness as a feature** | Fraudsters avoid identifiable devices — missing DeviceType has lift > 1.5× |
| **Velocity features** | Cumulative card transaction counts catch rapid small-test-then-large-fraud patterns |

---

## Feature Groups

| Group | Features | Source |
|-------|----------|--------|
| Time | `hour`, `day_of_week`, `is_night`, `is_weekend`, `is_business_hours` | `TransactionDT` decoded |
| Velocity | `card_txn_count_cumulative`, `time_since_last_txn`, `is_first_txn_for_card` | Per `card1`, sorted by time |
| Amount | `amount_z_score`, `amount_to_mean_ratio`, `log_amount`, card-level stats | Per `card1` aggregations |
| Interactions | `large_night_txn`, `high_zscore_night`, `first_txn_large_amt` | Combined weak signals |
| Missingness | `{col}_was_missing` flags | Columns with 10–80% missing |
| Categorical | Target-encoded `P_emaildomain`, `R_emaildomain`; label-encoded `ProductCD`, `card4`, `card6` | CV target encoding |
