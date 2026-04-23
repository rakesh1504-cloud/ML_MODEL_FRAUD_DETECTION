import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "distance_from_home_km",
    "num_transactions_last_24h",
]
BINARY_FEATURES = ["card_present", "is_foreign_transaction"]
CATEGORICAL_FEATURES = ["merchant_category"]
TARGET = "is_fraud"


class DataPreprocessor:
    """Clean, encode, and split transaction data."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders: dict = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)
        df = df.drop_duplicates(subset=["transaction_id"] if "transaction_id" in df.columns else None)
        df["amount"] = df["amount"].clip(lower=0)
        df["hour"] = df["hour"].clip(0, 23)
        df["day_of_week"] = df["day_of_week"].clip(0, 6)
        df["distance_from_home_km"] = df["distance_from_home_km"].clip(lower=0)
        df["num_transactions_last_24h"] = df["num_transactions_last_24h"].clip(lower=0)
        logger.info(f"Cleaned data: {before} → {len(df)} rows")
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cols = [c for c in NUMERIC_FEATURES if c in X.columns]
        if fit:
            X[cols] = self.imputer.fit_transform(X[cols])
            X[cols] = self.scaler.fit_transform(X[cols])
            self._fitted = True
        else:
            X[cols] = self.imputer.transform(X[cols])
            X[cols] = self.scaler.transform(X[cols])
        return X

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        drop = {"transaction_id", TARGET}
        return [c for c in df.columns if c not in drop]

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET], random_state=random_state
        )
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=relative_val, stratify=train_val[TARGET], random_state=random_state
        )
        logger.info(f"Split sizes — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
        return train, val, test

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        df = self.clean(df)
        df = self.encode_categoricals(df, fit=True)
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        y = df[TARGET]
        X = self.scale_features(X, fit=True)

        train_idx, val_idx, test_idx = self._split_indices(y)
        return (
            X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx],
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean(df)
        df = self.encode_categoricals(df, fit=False)
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        X = self.scale_features(X, fit=False)
        return X

    def _split_indices(self, y: pd.Series):
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        train_val, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
        train, val = train_test_split(train_val, test_size=0.125, stratify=y.iloc[train_val], random_state=42)
        return train, val, test

    def save(self, path: str = "data/processed/preprocessor.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str = "data/processed/preprocessor.pkl") -> "DataPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
