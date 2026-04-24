import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "isFraud"

# Columns that are identifiers or raw time offsets — never used as features
DROP_ALWAYS = {"TransactionID", "TransactionDT", TARGET}

# Missing-value thresholds (from EDA notebook section 4 & 9)
MISSING_DROP_THRESHOLD = 0.80   # drop columns with > 80% missing
MISSING_FLAG_THRESHOLD = 0.10   # create col_was_missing flag when > 10% missing

CATEGORICAL_FEATURES = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
MATCH_FEATURES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


class DataPreprocessor:
    """
    Clean, encode, and split IEEE-CIS transaction data.

    Missing-value strategy (from EDA):
      > 80% missing  → DROP column
      10–80% missing → add col_was_missing binary flag, then impute
      < 10% missing  → impute only
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.label_encoders: dict = {}

        # Columns learned during fit
        self.cols_to_drop: List[str] = []
        self.cols_to_flag: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        df = self._clean(df)
        df = self._handle_missing(df, fit=True)
        df = self._encode_match_features(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_numerics(df, fit=True)
        self._fitted = True

        X, y = self._split_X_y(df)
        return self._stratified_split(X, y, test_size, val_size, random_state)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        df = self._clean(df)
        df = self._handle_missing(df, fit=False)
        df = self._encode_match_features(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numerics(df, fit=False)
        X, _ = self._split_X_y(df)
        return X

    def save(self, path: str = "data/processed/preprocessor.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str = "data/processed/preprocessor.pkl") -> "DataPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)
        if "TransactionID" in df.columns:
            df = df.drop_duplicates(subset=["TransactionID"])
        df["TransactionAmt"] = df["TransactionAmt"].clip(lower=0)
        logger.info(f"Clean: {before} → {len(df)} rows")
        return df

    def _handle_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            missing_rate = df.isnull().mean()
            self.cols_to_drop = [
                c for c in missing_rate.index
                if missing_rate[c] > MISSING_DROP_THRESHOLD
                and c not in DROP_ALWAYS
            ]
            self.cols_to_flag = [
                c for c in missing_rate.index
                if MISSING_FLAG_THRESHOLD < missing_rate[c] <= MISSING_DROP_THRESHOLD
                and c not in DROP_ALWAYS
            ]
            logger.info(
                f"Missing strategy: dropping {len(self.cols_to_drop)} cols (>{MISSING_DROP_THRESHOLD:.0%} missing), "
                f"flagging {len(self.cols_to_flag)} cols (>{MISSING_FLAG_THRESHOLD:.0%} missing)"
            )

        # Add was_missing flags before dropping/imputing
        for col in self.cols_to_flag:
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isnull().astype(int)

        # Drop high-missing columns
        drop_present = [c for c in self.cols_to_drop if c in df.columns]
        df = df.drop(columns=drop_present)

        return df

    def _encode_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """M1–M9 are T/F/NaN strings — encode to 1/0/-1."""
        for col in MATCH_FEATURES:
            if col in df.columns:
                df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if fit:
            self.categorical_cols = cat_present

        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = (
                    df[col].astype(str).fillna("unknown")
                    .apply(lambda x: x if x in known else le.classes_[0])
                )
                df[col] = le.transform(df[col])
        return df

    def _scale_numerics(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        feature_cols = self._feature_columns(df)
        num_cols = [
            c for c in feature_cols
            if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]
            and c not in self.categorical_cols
        ]
        if fit:
            self.numeric_cols = num_cols

        cols_present = [c for c in self.numeric_cols if c in df.columns]
        if fit:
            df[cols_present] = self.num_imputer.fit_transform(df[cols_present])
            df[cols_present] = self.scaler.fit_transform(df[cols_present])
        else:
            df[cols_present] = self.num_imputer.transform(df[cols_present])
            df[cols_present] = self.scaler.transform(df[cols_present])
        return df

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in DROP_ALWAYS]

    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = self._feature_columns(df)
        y = df[TARGET] if TARGET in df.columns else None
        return df[feature_cols], y

    def _stratified_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        val_size: float,
        random_state: int,
    ):
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=relative_val, stratify=y_tv, random_state=random_state
        )
        logger.info(
            f"Split — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
