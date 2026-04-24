import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

TARGET = "isFraud"
DROP_ALWAYS = {"TransactionID", "TransactionDT", TARGET}

# Missing-value thresholds (notebook section 2 & 3)
MISSING_DROP_THRESHOLD = 0.80
MISSING_FLAG_THRESHOLD = 0.10


class DataPreprocessor:
    """
    Clean, impute, and split IEEE-CIS data for model training.

    Split strategy: TIME-BASED (notebook section 11 — not random).
      Fraud patterns evolve; train on the past, test on the future.
      Random splits allow future info to leak into training.

    Class imbalance: SMOTE applied ONLY on training data, AFTER the split
      (notebook section 12 — applying before split leaks synthetic fraud
      samples into the test set, the most common mistake on fraud projects).
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        smote_strategy: float = 0.1,
        smote_random_state: int = 42,
    ):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.smote_strategy = smote_strategy       # fraud becomes 10% of train
        self.smote_random_state = smote_random_state

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")

        self.cols_to_drop: List[str] = []
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        apply_smote: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns X_train (SMOTE-balanced), X_test, y_train, y_test.
        Note: only one test set — no separate validation here; use
        cross-validation inside ModelTrainer for hyperparameter tuning.
        """
        df = self._drop_high_missing(df, fit=True)
        df = self._impute(df, fit=True)
        self._fitted = True

        X, y = self._split_X_y(df)
        X_train, X_test, y_train, y_test = self._time_based_split(df, X, y, test_size)

        if apply_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        logger.info(
            f"Final sets — train: {len(X_train):,}  test: {len(X_test):,}  "
            f"train fraud rate: {y_train.mean()*100:.2f}%  "
            f"test fraud rate: {y_test.mean()*100:.2f}%"
        )
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        df = df.copy()
        drop_present = [c for c in self.cols_to_drop if c in df.columns]
        df = df.drop(columns=drop_present)
        df = self._impute(df, fit=False)
        X, _ = self._split_X_y(df)
        return X

    def save_parquet(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Save processed datasets to Parquet (10x smaller than CSV, preserves dtypes)."""
        X_train.to_parquet(self.processed_dir / "X_train.parquet", index=False)
        y_train.to_frame().to_parquet(self.processed_dir / "y_train.parquet", index=False)
        X_test.to_parquet(self.processed_dir / "X_test.parquet", index=False)
        y_test.to_frame().to_parquet(self.processed_dir / "y_test.parquet", index=False)
        logger.info("Saved X_train / y_train / X_test / y_test to data/processed/ (Parquet)")

    def load_parquet(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train = pd.read_parquet(self.processed_dir / "X_train.parquet")
        y_train = pd.read_parquet(self.processed_dir / "y_train.parquet").squeeze()
        X_test  = pd.read_parquet(self.processed_dir / "X_test.parquet")
        y_test  = pd.read_parquet(self.processed_dir / "y_test.parquet").squeeze()
        return X_train, X_test, y_train, y_test

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

    def _drop_high_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            missing_rate = df.isnull().mean()
            self.cols_to_drop = [
                c for c in missing_rate.index
                if missing_rate[c] > MISSING_DROP_THRESHOLD
                and c not in DROP_ALWAYS
            ]
            logger.info(f"Dropping {len(self.cols_to_drop)} cols with >{MISSING_DROP_THRESHOLD:.0%} missing")
        drop_present = [c for c in self.cols_to_drop if c in df.columns]
        return df.drop(columns=drop_present)

    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]

        if fit:
            self.numeric_cols = [
                c for c in feature_cols
                if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, int, float]
            ]
            self.cat_cols = [
                c for c in feature_cols
                if df[c].dtype == object and c not in self.numeric_cols
            ]

        num_present = [c for c in self.numeric_cols if c in df.columns]
        cat_present = [c for c in self.cat_cols if c in df.columns]

        if num_present:
            if fit:
                df[num_present] = self.num_imputer.fit_transform(df[num_present])
            else:
                df[num_present] = self.num_imputer.transform(df[num_present])

        if cat_present:
            if fit:
                df[cat_present] = self.cat_imputer.fit_transform(df[cat_present])
            else:
                df[cat_present] = self.cat_imputer.transform(df[cat_present])

        remaining = df[feature_cols].isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"{remaining} nulls remain after imputation")
        return df

    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
        y = df[TARGET] if TARGET in df.columns else None
        return df[feature_cols], y

    def _time_based_split(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Sort by TransactionDT, take first 80% as train, last 20% as test.
        This prevents future data leaking into training (notebook section 11).
        """
        if "TransactionDT" in df.columns:
            sorted_idx = df["TransactionDT"].argsort().values
        else:
            sorted_idx = np.arange(len(df))

        split_idx = int(len(sorted_idx) * (1 - test_size))
        train_idx = sorted_idx[:split_idx]
        test_idx  = sorted_idx[split_idx:]

        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

        train_max = df["TransactionDT"].iloc[train_idx].max() if "TransactionDT" in df.columns else None
        test_min  = df["TransactionDT"].iloc[test_idx].min() if "TransactionDT" in df.columns else None
        logger.info(
            f"Time-based split — train: {len(X_train):,} | test: {len(X_test):,} | "
            f"no leakage: {test_min >= train_max if train_max and test_min else 'N/A'}"
        )
        return X_train, X_test, y_train, y_test

    def _apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        SMOTE only on training data, AFTER the split (notebook section 12).
        sampling_strategy=0.1 → fraud becomes 10% of training set.
        """
        logger.info(
            f"Before SMOTE — fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)"
        )
        smote = SMOTE(
            random_state=self.smote_random_state,
            sampling_strategy=self.smote_strategy,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=TARGET)
        logger.info(
            f"After  SMOTE — fraud: {y_res.sum():,} ({y_res.mean()*100:.2f}%)"
        )
        return X_res, y_res
