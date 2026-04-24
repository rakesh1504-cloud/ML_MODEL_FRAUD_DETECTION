import pandas as pd
import numpy as np
import logging
from typing import List

from src.data.ingestion import IEEE_REF_DATE

logger = logging.getLogger(__name__)

# High-risk product codes identified in EDA section 7
_HIGH_RISK_PRODUCT = {"C", "S"}
# High-risk email domains (anonymous / temporary)
_HIGH_RISK_EMAIL = {"anonymous.com", "protonmail.com", "guerrillamail.com"}


class FeatureEngineer:
    """
    Derive additional features from raw IEEE-CIS columns.

    Transformations are based on EDA findings:
      - Time patterns    : hour, day_of_week, is_night, is_weekend (sections 6)
      - Amount patterns  : log_amount, is_round_amount (section 5)
      - Risk signals     : high-risk product/email, missingness lift (sections 7, 9)
      - Velocity proxy   : C1 (transaction count feature from dataset)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._time_features(df)
        df = self._amount_features(df)
        df = self._risk_features(df)
        logger.info(f"Feature engineering complete — {df.shape[1]} total columns")
        return df

    # ------------------------------------------------------------------

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decode TransactionDT (seconds offset) into human-readable time features."""
        if "TransactionDT" not in df.columns:
            return df

        dt_series = IEEE_REF_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")

        if "hour" not in df.columns:
            df["hour"] = dt_series.dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt_series.dt.dayofweek

        # Night = 11pm–6am (elevated fraud, EDA section 6)
        df["is_night"] = df["hour"].isin(list(range(0, 6)) + [23]).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclic encoding prevents hour-0 / hour-23 distance artefacts
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """EDA section 5: fraud transactions have higher and more varied amounts."""
        if "TransactionAmt" not in df.columns:
            return df

        df["log_amount"] = np.log1p(df["TransactionAmt"])
        df["is_large_transaction"] = (
            df["TransactionAmt"] > df["TransactionAmt"].quantile(0.95)
        ).astype(int)
        df["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(int)
        return df

    def _risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """EDA sections 7 & 9: categorical and missingness-based risk signals."""

        # Product code risk (EDA section 7)
        if "ProductCD" in df.columns:
            df["is_high_risk_product"] = (
                df["ProductCD"].astype(str).isin(_HIGH_RISK_PRODUCT)
            ).astype(int)

        # Email domain risk (anonymous.com is a strong fraud signal)
        if "P_emaildomain" in df.columns:
            df["is_risky_email"] = (
                df["P_emaildomain"].astype(str).isin(_HIGH_RISK_EMAIL)
            ).astype(int)

        # Recipient ≠ purchaser email → potential account takeover signal
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            df["email_domain_mismatch"] = (
                df["P_emaildomain"].astype(str) != df["R_emaildomain"].astype(str)
            ).astype(int)

        # Device missingness = fraud signal (EDA section 9, lift > 1.5)
        if "DeviceType_was_missing" not in df.columns and "DeviceType" in df.columns:
            df["device_missing"] = df["DeviceType"].isnull().astype(int)

        # High-velocity proxy: C1 = number of payment methods found (IEEE-CIS)
        if "C1" in df.columns:
            df["high_velocity"] = (df["C1"] > 5).astype(int)

        return df

    def feature_names(self, base_columns: List[str]) -> List[str]:
        extra = [
            "hour", "day_of_week",
            "is_night", "is_weekend",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "log_amount", "is_large_transaction", "is_round_amount",
            "is_high_risk_product", "is_risky_email", "email_domain_mismatch",
            "device_missing", "high_velocity",
        ]
        return base_columns + [f for f in extra if f not in base_columns]
