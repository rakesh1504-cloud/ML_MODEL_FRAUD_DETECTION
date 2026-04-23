import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Derive additional features from raw transaction columns."""

    # Hour buckets: night / morning / afternoon / evening
    _HOUR_BUCKETS = [(0, 6, "night"), (6, 12, "morning"), (12, 18, "afternoon"), (18, 24, "evening")]

    # High-risk merchant categories (based on fraud literature)
    _HIGH_RISK_CATEGORIES = {"online", "travel", "entertainment"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._time_features(df)
        df = self._amount_features(df)
        df = self._risk_features(df)
        df = self._velocity_features(df)
        logger.info(f"Feature engineering complete — {df.shape[1]} total columns")
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "hour" not in df.columns:
            return df
        df["is_night"] = df["hour"].between(0, 5).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int) if "day_of_week" in df.columns else 0
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        if "day_of_week" in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "amount" not in df.columns:
            return df
        df["log_amount"] = np.log1p(df["amount"])
        df["is_large_transaction"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
        df["is_round_amount"] = (df["amount"] % 10 == 0).astype(int)
        return df

    def _risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "merchant_category" in df.columns:
            # Works on both string and encoded integer form
            df["is_high_risk_category"] = df["merchant_category"].apply(
                lambda x: int(str(x).lower() in self._HIGH_RISK_CATEGORIES)
            )
        if "distance_from_home_km" in df.columns:
            df["is_far_from_home"] = (df["distance_from_home_km"] > 100).astype(int)
        combo_flags = []
        if "is_foreign_transaction" in df.columns:
            combo_flags.append("is_foreign_transaction")
        if "is_far_from_home" in df.columns:
            combo_flags.append("is_far_from_home")
        if len(combo_flags) == 2:
            df["geo_risk_score"] = df[combo_flags].sum(axis=1)
        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "num_transactions_last_24h" not in df.columns:
            return df
        df["high_velocity"] = (df["num_transactions_last_24h"] > 5).astype(int)
        df["extreme_velocity"] = (df["num_transactions_last_24h"] > 10).astype(int)
        return df

    def feature_names(self, base_columns: List[str]) -> List[str]:
        """Return expected column names after transform (for documentation)."""
        extra = [
            "is_night", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "log_amount", "is_large_transaction", "is_round_amount",
            "is_high_risk_category", "is_far_from_home", "geo_risk_score",
            "high_velocity", "extreme_velocity",
        ]
        return base_columns + [f for f in extra if f not in base_columns]
