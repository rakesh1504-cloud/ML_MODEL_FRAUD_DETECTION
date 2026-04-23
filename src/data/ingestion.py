import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class DataIngestion:
    """Load transaction data from CSV, Parquet, or synthetic generation."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        logger.info(f"Loading CSV from {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        logger.info(f"Loading Parquet from {path}")
        df = pd.read_parquet(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def load_file(self, filename: str, **kwargs) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        ext = path.suffix.lower()
        if ext == ".csv":
            return self.load_csv(filename, **kwargs)
        elif ext in (".parquet", ".pq"):
            return self.load_parquet(filename, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def generate_synthetic(
        self,
        n_samples: int = 10_000,
        fraud_rate: float = 0.02,
        random_state: int = 42,
        save: bool = True,
    ) -> pd.DataFrame:
        """Generate a synthetic credit-card-style transaction dataset."""
        rng = np.random.default_rng(random_state)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        def _make_transactions(n: int, is_fraud: bool) -> dict:
            label = int(is_fraud)
            multiplier = 5.0 if is_fraud else 1.0
            return {
                "transaction_id": [f"TXN{i:08d}" for i in rng.integers(0, 10**8, n)],
                "amount": np.abs(rng.normal(200 * multiplier, 100 * multiplier, n)),
                "hour": rng.integers(0, 24, n),
                "day_of_week": rng.integers(0, 7, n),
                "merchant_category": rng.choice(
                    ["grocery", "travel", "online", "entertainment", "gas"], n
                ),
                "card_present": rng.choice([0, 1], n, p=[0.6, 0.4] if is_fraud else [0.1, 0.9]),
                "distance_from_home_km": np.abs(rng.normal(500 if is_fraud else 10, 200, n)),
                "num_transactions_last_24h": rng.integers(1, 20 if is_fraud else 5, n),
                "is_foreign_transaction": rng.choice([0, 1], n, p=[0.3, 0.7] if is_fraud else [0.95, 0.05]),
                "is_fraud": np.full(n, label),
            }

        legit = pd.DataFrame(_make_transactions(n_legit, False))
        fraud = pd.DataFrame(_make_transactions(n_fraud, True))
        df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=random_state)
        df = df.reset_index(drop=True)

        if save:
            out_path = self.raw_data_dir / "transactions.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Synthetic dataset saved to {out_path}")

        logger.info(
            f"Generated {n_samples:,} transactions — fraud: {n_fraud:,} ({fraud_rate:.1%})"
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        required = {
            "amount", "hour", "day_of_week", "merchant_category",
            "card_present", "distance_from_home_km",
            "num_transactions_last_24h", "is_foreign_transaction",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
