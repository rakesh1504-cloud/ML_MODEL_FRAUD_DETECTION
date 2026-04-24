import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reference timestamp used by the IEEE-CIS dataset for TransactionDT
IEEE_REF_DATE = pd.Timestamp("2017-12-01")

REQUIRED_COLUMNS = {
    "TransactionID",
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "isFraud",
}


class DataIngestion:
    """Load and merge IEEE-CIS transaction + identity data, or generate synthetic equivalents."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_ieee_cis(
        self,
        txn_file: str = "train_transaction.csv",
        identity_file: str = "train_identity.csv",
    ) -> pd.DataFrame:
        """Load and left-join IEEE-CIS transaction + identity files."""
        txn_path = self.raw_data_dir / txn_file
        ident_path = self.raw_data_dir / identity_file

        logger.info(f"Loading transactions from {txn_path}")
        txn = pd.read_csv(txn_path)
        logger.info(f"  transactions shape: {txn.shape}")

        if ident_path.exists():
            logger.info(f"Loading identity from {ident_path}")
            ident = pd.read_csv(ident_path)
            logger.info(f"  identity shape: {ident.shape}")
            df = txn.merge(ident, on="TransactionID", how="left")
        else:
            logger.warning(f"Identity file not found at {ident_path} — using transactions only")
            df = txn

        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info(f"Merged: {df.shape[0]:,} rows x {df.shape[1]} cols  |  {mem_mb:.0f} MB")
        return df

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

    # ------------------------------------------------------------------
    # Synthetic data (mirrors IEEE-CIS schema)
    # ------------------------------------------------------------------

    def generate_synthetic(
        self,
        n_samples: int = 10_000,
        fraud_rate: float = 0.035,
        random_state: int = 42,
        save: bool = True,
    ) -> pd.DataFrame:
        """Generate a synthetic dataset that mirrors the IEEE-CIS schema."""
        rng = np.random.default_rng(random_state)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        def _make(n: int, is_fraud: bool) -> dict:
            mult = 5.0 if is_fraud else 1.0
            # TransactionDT: seconds offset from IEEE_REF_DATE (~180 days window)
            dt = rng.integers(0, 180 * 24 * 3600, n)
            return {
                "TransactionID": rng.integers(2_987_000, 3_987_000, n),
                "TransactionDT": dt,
                "TransactionAmt": np.abs(rng.lognormal(4.5 * mult, 1.2, n)).clip(0.5, 20_000),
                "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
                "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n),
                "card6": rng.choice(["debit", "credit"], n),
                "P_emaildomain": rng.choice(
                    ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], n,
                    p=[0.05, 0.05, 0.05, 0.75, 0.10] if is_fraud else [0.35, 0.25, 0.20, 0.05, 0.15],
                ),
                "R_emaildomain": rng.choice(
                    ["gmail.com", "yahoo.com", "hotmail.com", np.nan], n
                ),
                "addr1": rng.integers(100, 500, n).astype(float),
                "addr2": rng.integers(10, 100, n).astype(float),
                "dist1": np.abs(rng.normal(500 if is_fraud else 10, 200, n)),
                "dist2": rng.choice([np.nan, *rng.integers(0, 3000, n).tolist()], n),
                "C1": rng.integers(1, 20 if is_fraud else 5, n).astype(float),
                "C2": rng.integers(1, 15, n).astype(float),
                "D1": rng.integers(0, 300, n).astype(float),
                "M1": rng.choice(["T", "F", np.nan], n),
                "M2": rng.choice(["T", "F", np.nan], n),
                "M3": rng.choice(["T", "F", np.nan], n),
                "DeviceType": rng.choice(
                    ["desktop", "mobile", np.nan], n,
                    p=[0.1, 0.1, 0.8] if is_fraud else [0.5, 0.4, 0.1],
                ),
                "isFraud": np.full(n, int(is_fraud)),
            }

        legit = pd.DataFrame(_make(n_legit, False))
        fraud = pd.DataFrame(_make(n_fraud, True))
        df = (
            pd.concat([legit, fraud], ignore_index=True)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

        if save:
            out = self.raw_data_dir / "transactions.csv"
            df.to_csv(out, index=False)
            logger.info(f"Synthetic dataset saved to {out}")

        imbalance = n_legit // max(n_fraud, 1)
        logger.info(
            f"Generated {n_samples:,} transactions — fraud: {n_fraud:,} ({fraud_rate:.1%}) "
            f"| ratio 1:{imbalance} | scale_pos_weight ≈ {imbalance}"
        )
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> bool:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
