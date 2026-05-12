import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Run: pip install optuna")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class OptunaHyperparamTuner:
    """
    Bayesian hyperparameter tuning via Optuna.

    Optimises AUC-PR (primary metric) on the test/validation set.
    Matches notebook section 5 — 50 trials finds near-optimal params.

    Usage:
        tuner = OptunaHyperparamTuner(X_train, y_train, X_val, y_val)
        best_params = tuner.tune_lightgbm(n_trials=50)
        best_model  = tuner.best_model
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Install optuna: pip install optuna")
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params: Dict = {}
        self.best_value: float = 0.0
        self.best_model: Optional[Any] = None
        self.study: Optional[Any] = None

    # ------------------------------------------------------------------
    # LightGBM tuning (notebook section 5)
    # ------------------------------------------------------------------

    def tune_lightgbm(self, n_trials: int = 50, study_name: str = "lgbm_fraud_aucpr") -> Dict:
        if not _LGBM_AVAILABLE:
            raise ImportError("Install lightgbm: pip install lightgbm")

        logger.info(f"Optuna: tuning LightGBM with {n_trials} trials (optimising AUC-PR)...")

        self.study = optuna.create_study(direction="maximize", study_name=study_name)
        self.study.optimize(self._lgbm_objective, n_trials=n_trials, show_progress_bar=False)

        self.best_value = self.study.best_value
        self.best_params = self.study.best_params.copy()
        self.best_params.update({
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        })

        logger.info(f"Best AUC-PR: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Re-train champion on full data with best params
        self.best_model = LGBMClassifier(**self.best_params)
        self.best_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        return self.best_params

    def _lgbm_objective(self, trial: Any) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        prob = model.predict_proba(self.X_val)[:, 1]
        return average_precision_score(self.y_val, prob)

    # ------------------------------------------------------------------
    # XGBoost tuning
    # ------------------------------------------------------------------

    def tune_xgboost(self, n_trials: int = 50, study_name: str = "xgb_fraud_aucpr") -> Dict:
        if not _XGB_AVAILABLE:
            raise ImportError("Install xgboost: pip install xgboost")

        n_legit = int((self.y_train == 0).sum())
        n_fraud = int(self.y_train.sum())
        self._spw = round(n_legit / max(n_fraud, 1), 1)

        logger.info(f"Optuna: tuning XGBoost with {n_trials} trials (scale_pos_weight={self._spw})...")
        self.study = optuna.create_study(direction="maximize", study_name=study_name)
        self.study.optimize(self._xgb_objective, n_trials=n_trials, show_progress_bar=False)

        self.best_value = self.study.best_value
        self.best_params = self.study.best_params.copy()
        self.best_params.update({
            "scale_pos_weight": self._spw,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        })
        logger.info(f"Best XGBoost AUC-PR: {self.best_value:.4f}")

        self.best_model = XGBClassifier(**self.best_params)
        self.best_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )
        return self.best_params

    def _xgb_objective(self, trial: Any) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": self._spw,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 30,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )
        prob = model.predict_proba(self.X_val)[:, 1]
        return average_precision_score(self.y_val, prob)
