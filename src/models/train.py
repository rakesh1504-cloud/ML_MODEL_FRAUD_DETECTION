import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# LightGBM is optional — used in notebook section 14 for feature importance preview
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Run: pip install lightgbm")

SUPPORTED_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}
if _LGBM_AVAILABLE:
    SUPPORTED_MODELS["lightgbm"] = LGBMClassifier

DEFAULT_PARAMS: Dict[str, Dict] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": 42,
    },
    # LightGBM defaults match notebook section 14 quick-check params
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",   # handles imbalance even after SMOTE
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    },
}


class ModelTrainer:
    """
    Train, persist, and reload fraud detection classifiers.

    Supports: logistic_regression, random_forest, gradient_boosting, lightgbm.

    For gradient_boosting, pass scale_pos_weight in params to apply sample
    weights equivalent to XGBoost/LightGBM scale_pos_weight behaviour.
    (With SMOTE the imbalance is already reduced to ~1:9, so this is less critical.)
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.training_metadata: Dict = {}
        self._scale_pos_weight: Optional[float] = None

    def build(self, model_name: str = "lightgbm", params: Optional[Dict] = None) -> Any:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(SUPPORTED_MODELS)}"
            )
        merged = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
        self._scale_pos_weight = merged.pop("scale_pos_weight", None)

        self.model = SUPPORTED_MODELS[model_name](**merged)
        self.model_name = model_name
        logger.info(f"Built {model_name} | params: {merged}")
        return self.model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "ModelTrainer":
        if self.model is None:
            raise RuntimeError("Call build() before fit()")

        sample_weight = None
        if self._scale_pos_weight and self.model_name == "gradient_boosting":
            sample_weight = np.where(y_train == 1, self._scale_pos_weight, 1.0)

        logger.info(f"Training {self.model_name} on {len(X_train):,} samples …")
        t0 = time.time()

        fit_kwargs: Dict = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self.model.fit(X_train, y_train, **fit_kwargs)
        elapsed = time.time() - t0

        self.training_metadata = {
            "model_name": self.model_name,
            "n_train": len(X_train),
            "n_features": X_train.shape[1],
            "train_time_s": round(elapsed, 2),
            "fraud_rate_train": round(float(y_train.mean()), 4),
            "scale_pos_weight": self._scale_pos_weight,
        }

        if X_val is not None and y_val is not None:
            from sklearn.metrics import average_precision_score, roc_auc_score
            val_proba = self.model.predict_proba(X_val)[:, 1]
            self.training_metadata["val_auc_pr"] = round(average_precision_score(y_val, val_proba), 4)
            self.training_metadata["val_roc_auc"] = round(roc_auc_score(y_val, val_proba), 4)
            logger.info(
                f"Validation — AUC-PR: {self.training_metadata['val_auc_pr']:.4f}  "
                f"ROC-AUC: {self.training_metadata['val_roc_auc']:.4f}"
            )

        logger.info(f"Training complete in {elapsed:.1f}s")
        return self

    def feature_importances(self, feature_names: list) -> pd.DataFrame:
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(f"{self.model_name} does not expose feature_importances_")
        return (
            pd.DataFrame({"feature": feature_names, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, filename: Optional[str] = None) -> str:
        if self.model is None:
            raise RuntimeError("No model to save. Run fit() first.")
        filename = filename or f"{self.model_name}.pkl"
        path = self.model_dir / filename
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "metadata": self.training_metadata}, f)
        logger.info(f"Model saved to {path}")
        return str(path)

    def load(self, filename: str) -> "ModelTrainer":
        path = self.model_dir / filename
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.training_metadata = payload.get("metadata", {})
        self.model_name = self.training_metadata.get("model_name", filename)
        logger.info(f"Model loaded from {path}")
        return self
