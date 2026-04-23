import pickle
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

DEFAULT_PARAMS: Dict[str, Dict] = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": 42},
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
}


class ModelTrainer:
    """Train, persist, and reload fraud detection classifiers."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.training_metadata: Dict = {}

    def build(self, model_name: str = "random_forest", params: Optional[Dict] = None) -> Any:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(SUPPORTED_MODELS)}")
        cls = SUPPORTED_MODELS[model_name]
        merged_params = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
        self.model = cls(**merged_params)
        self.model_name = model_name
        logger.info(f"Built {model_name} with params: {merged_params}")
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
        logger.info(f"Training {self.model_name} on {len(X_train):,} samples …")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - t0
        self.training_metadata = {
            "model_name": self.model_name,
            "n_train": len(X_train),
            "n_features": X_train.shape[1],
            "train_time_s": round(elapsed, 2),
            "fraud_rate_train": float(y_train.mean()),
        }
        if X_val is not None and y_val is not None:
            from sklearn.metrics import roc_auc_score
            val_proba = self.model.predict_proba(X_val)[:, 1]
            self.training_metadata["val_roc_auc"] = round(roc_auc_score(y_val, val_proba), 4)
            logger.info(f"Validation ROC-AUC: {self.training_metadata['val_roc_auc']:.4f}")
        logger.info(f"Training complete in {elapsed:.1f}s")
        return self

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

    def feature_importances(self, feature_names: list) -> pd.DataFrame:
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(f"{self.model_name} does not expose feature_importances_")
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
