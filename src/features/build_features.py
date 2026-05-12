import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from src.data.ingestion import IEEE_REF_DATE

logger = logging.getLogger(__name__)

# Categorical encoding strategy (notebook section 9)
HIGH_CARD_COLS = ["P_emaildomain", "R_emaildomain", "card1"]   # target encode
LOW_CARD_COLS  = ["ProductCD", "card4", "card6"]                # label encode


class FeatureEngineer:
    """
    Full feature engineering pipeline based on 01_feature_engineering.ipynb.

    Features built (in order):
      1. Missingness flags      — col_was_missing for 10-80% missing cols
      2. Time features          — hour, day_of_week, day_of_month, month,
                                  is_weekend, is_night, is_business_hours
      3. Velocity features      — per-card cumulative counts / amounts / time deltas
      4. Amount aggregations    — per-card mean/std/z-score/ratio
      5. Interaction features   — large_night_txn, high_zscore_night, etc.
      6. Categorical encoding   — target encode (CV) high-card; label encode low-card
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self._target_means: Dict[str, Dict] = {}   # col -> {category: mean}
        self._global_means: Dict[str, float] = {}  # col -> overall fraud mean
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._flag_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all features during training (fits encoders, CV target encode)."""
        df = df.copy()
        df = self._missingness_flags(df, fit=True)
        df = self._time_features(df)
        df = self._sort_by_time(df)
        df = self._velocity_features(df)
        df = self._amount_aggregations(df)
        df = self._interaction_features(df)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        logger.info(f"Feature engineering (fit) complete — {df.shape[1]} columns")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all features during inference (uses fitted encoders)."""
        df = df.copy()
        df = self._missingness_flags(df, fit=False)
        df = self._time_features(df)
        df = self._sort_by_time(df)
        df = self._velocity_features(df)
        df = self._amount_aggregations(df)
        df = self._interaction_features(df)
        df = self._encode_categoricals(df, fit=False)
        logger.info(f"Feature engineering (transform) complete — {df.shape[1]} columns")
        return df

    def save_feature_cols(self, feature_cols: List[str], path: str = "data/processed/feature_cols.json") -> None:
        with open(path, "w") as f:
            json.dump(feature_cols, f)
        logger.info(f"Feature columns saved to {path}")

    @staticmethod
    def load_feature_cols(path: str = "data/processed/feature_cols.json") -> List[str]:
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Step 1 — Missingness flags (notebook section 3)
    # ------------------------------------------------------------------

    def _missingness_flags(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            missing_rate = df.isnull().mean()
            self._flag_cols = [
                c for c in missing_rate.index
                if 0.10 < missing_rate[c] < 0.80
                and c not in ("isFraud", "TransactionID", "TransactionDT")
            ]
            logger.info(f"Missingness flags: {len(self._flag_cols)} columns")

        for col in self._flag_cols:
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isnull().astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Step 2 — Time features (notebook section 5)
    # ------------------------------------------------------------------

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionDT" not in df.columns:
            return df

        dt = IEEE_REF_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
        df["hour"]             = dt.dt.hour.astype(np.int8)
        df["day_of_week"]      = dt.dt.dayofweek.astype(np.int8)
        df["day_of_month"]     = dt.dt.day.astype(np.int8)
        df["month"]            = dt.dt.month.astype(np.int8)
        df["is_weekend"]       = (df["day_of_week"] >= 5).astype(np.int8)
        df["is_night"]         = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(np.int8)
        df["is_business_hours"] = (
            (df["hour"] >= 9) & (df["hour"] <= 17) & (df["is_weekend"] == 0)
        ).astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Step 3 — Sort by time (required for leak-free velocity)
    # ------------------------------------------------------------------

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Step 4 — Velocity features per card (notebook section 6)
    # ------------------------------------------------------------------

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "card1" not in df.columns:
            return df

        df["card_txn_count_cumulative"] = (
            df.groupby("card1").cumcount() + 1
        ).astype(np.int32)

        if "TransactionAmt" in df.columns:
            df["card_amt_cumulative"] = (
                df.groupby("card1")["TransactionAmt"].cumsum()
            ).astype(np.float32)

        if "TransactionDT" in df.columns:
            df["time_since_last_txn"] = (
                df.groupby("card1")["TransactionDT"].diff().fillna(0)
            ).astype(np.float32)

        df["is_first_txn_for_card"] = (
            df["card_txn_count_cumulative"] == 1
        ).astype(np.int8)

        return df

    # ------------------------------------------------------------------
    # Step 5 — Amount aggregation features per card (notebook section 7)
    # ------------------------------------------------------------------

    def _amount_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionAmt" not in df.columns or "card1" not in df.columns:
            return df

        card_stats = (
            df.groupby("card1")["TransactionAmt"]
            .agg(["mean", "std", "min", "max", "median"])
            .rename(columns={
                "mean": "card_amt_mean", "std": "card_amt_std",
                "min": "card_amt_min", "max": "card_amt_max",
                "median": "card_amt_median",
            })
        )
        df = df.merge(card_stats, on="card1", how="left")
        df["card_amt_std"] = df["card_amt_std"].fillna(0)

        df["amount_z_score"] = (
            (df["TransactionAmt"] - df["card_amt_mean"])
            / (df["card_amt_std"] + 1e-6)
        ).astype(np.float32)

        df["amount_to_mean_ratio"] = (
            df["TransactionAmt"] / (df["card_amt_mean"] + 1e-6)
        ).astype(np.float32)

        df["log_amount"] = np.log1p(df["TransactionAmt"]).astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Step 6 — Interaction features (notebook section 8)
    # ------------------------------------------------------------------

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "is_night" not in df.columns or "TransactionAmt" not in df.columns:
            return df

        card_mean = df.get("card_amt_mean", pd.Series(0, index=df.index))

        df["large_night_txn"] = (
            (df["is_night"] == 1) & (df["TransactionAmt"] > card_mean * 2)
        ).astype(np.int8)

        df["first_txn_large_amt"] = (
            (df.get("is_first_txn_for_card", 0) == 1) &
            (df["TransactionAmt"] > 100)
        ).astype(np.int8)

        if "log_amount" in df.columns:
            df["amount_x_hour"] = (
                df["log_amount"] * df["hour"]
            ).astype(np.float32)

        if "amount_z_score" in df.columns:
            df["high_zscore_night"] = (
                (df["amount_z_score"] > 2) & (df["is_night"] == 1)
            ).astype(np.int8)

        return df

    # ------------------------------------------------------------------
    # Step 7 — Categorical encoding (notebook section 9)
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        # Target encode high-cardinality columns
        high_card = [c for c in HIGH_CARD_COLS if c in df.columns]
        for col in high_card:
            if fit:
                df[f"{col}_te"] = self._target_encode_cv(df, col)
            else:
                means = self._target_means.get(col, {})
                global_mean = self._global_means.get(col, 0.0)
                df[f"{col}_te"] = (
                    df[col].astype(str).map(means).fillna(global_mean)
                ).astype(np.float32)

        # Label encode low-cardinality columns
        low_card = [c for c in LOW_CARD_COLS if c in df.columns]
        for col in low_card:
            if fit:
                le = LabelEncoder()
                df[f"{col}_le"] = le.fit_transform(df[col].astype(str).fillna("missing"))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[f"{col}_le"] = le.transform(
                        df[col].astype(str).fillna("missing")
                            .apply(lambda x: x if x in known else le.classes_[0])
                    )

        # Drop original string columns
        df = df.drop(columns=high_card + low_card, errors="ignore")
        return df

    def _target_encode_cv(self, df: pd.DataFrame, col: str, target: str = "isFraud") -> pd.Series:
        """CV target encoding to prevent data leakage (notebook section 9)."""
        encoded = pd.Series(np.nan, index=df.index, dtype=float)
        global_mean = float(df[target].mean())
        self._global_means[col] = global_mean
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for train_idx, val_idx in kf.split(df):
            fold_means = df.iloc[train_idx].groupby(col)[target].mean()
            encoded.iloc[val_idx] = (
                df.iloc[val_idx][col].astype(str).map(fold_means).fillna(global_mean)
            )

        # Store global means for inference
        self._target_means[col] = df.groupby(col)[target].mean().astype(float).to_dict()
        return encoded
