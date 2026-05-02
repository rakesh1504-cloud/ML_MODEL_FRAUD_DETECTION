import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Run: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Run: pip install xgboost")

SUPPORTED_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}
if _LGBM_AVAILABLE:
    SUPPORTED_MODELS["lightgbm"] = LGBMClassifier
if _XGB_AVAILABLE:
    SUPPORTED_MODELS["xgboost"] = XGBClassifier

DEFAULT_PARAMS: Dict[str, Dict] = {
    "logistic_regression": {
        "C": 1.0, "max_iter": 1000,
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5,
        "class_weight": "balanced", "n_jobs": -1, "random_state": 42,
    },
    "gradient_boosting": {
        "n_estimators": 200, "learning_rate": 0.05,
        "max_depth": 5, "subsample": 0.8, "random_state": 42,
    },
    # Matches notebook section 3 untuned defaults
    "lightgbm": {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 8, "num_leaves": 63,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 20, "class_weight": "balanced",
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    },
    # Matches notebook section 4 untuned defaults
    "xgboost": {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "aucpr", "early_stopping_rounds": 50,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    },
}


class ModelTrainer:
    """
    Train, persist, and reload fraud detection classifiers.

    Supports: logistic_regression, random_forest, gradient_boosting,
              lightgbm (with early stopping), xgboost (with scale_pos_weight).

    scale_pos_weight is auto-computed from y_train for XGBoost if not set.
    For gradient_boosting it is applied as sample_weight during fit.
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
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(SUPPORTED_MODELS)}")
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

        fit_kwargs: Dict = {}

        # XGBoost: auto-compute scale_pos_weight from class ratio
        if self.model_name == "xgboost":
            spw = self._scale_pos_weight
            if spw is None:
                n_legit = int((y_train == 0).sum())
                n_fraud = int(y_train.sum())
                spw = round(n_legit / max(n_fraud, 1), 1)
                logger.info(f"XGBoost scale_pos_weight auto-set to {spw}")
            self.model.set_params(scale_pos_weight=spw)
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False

        # LightGBM: early stopping via callbacks
        if self.model_name == "lightgbm" and _LGBM_AVAILABLE:
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]

        # GradientBoosting: scale_pos_weight as sample_weight
        if self.model_name == "gradient_boosting" and self._scale_pos_weight:
            fit_kwargs["sample_weight"] = np.where(
                y_train == 1, self._scale_pos_weight, 1.0
            )

        logger.info(f"Training {self.model_name} on {len(X_train):,} samples …")
        t0 = time.time()
        self.model.fit(X_train, y_train, **fit_kwargs)
        elapsed = time.time() - t0

        self.training_metadata = {
            "model_name": self.model_name,
            "n_train": len(X_train),
            "n_features": X_train.shape[1],
            "train_time_s": round(elapsed, 2),
            "fraud_rate_train": round(float(y_train.mean()), 4),
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

    def feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
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

    def save_champion(self, path: str = "models/lgbm_champion.pkl") -> str:
        """Save champion model to the top-level models/ dir (notebook section 12)."""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Champion model saved to {path}")
        return path

    def load(self, filename: str) -> "ModelTrainer":
        path = self.model_dir / filename
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.training_metadata = payload.get("metadata", {})
        self.model_name = self.training_metadata.get("model_name", filename)
        logger.info(f"Model loaded from {path}")
        return self
