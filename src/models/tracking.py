import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Run: pip install mlflow")


class MLflowTracker:
    """
    MLflow experiment tracking for the fraud detection pipeline.

    Matches notebook section 13 — logs params, metrics, model, and artefacts.
    Supports local tracking (default) or DagsHub remote tracking URI.

    Usage:
        tracker = MLflowTracker(experiment="fraud-detection-v1")
        tracker.log_run(
            model=lgbm_tuned,
            params=best_params,
            metrics=test_metrics,
            threshold_data=threshold_data,
            artefact_paths=["data/external/shap_beeswarm.png"],
        )
    """

    def __init__(
        self,
        experiment: str = "fraud-detection-v1",
        tracking_uri: Optional[str] = None,
        dagshub_user: Optional[str] = None,
        dagshub_repo: str = "fraud-detection",
    ):
        if not _MLFLOW_AVAILABLE:
            raise ImportError("Install mlflow: pip install mlflow")

        if dagshub_user:
            uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        elif tracking_uri:
            uri = tracking_uri
        else:
            uri = "mlruns"  # local default

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self.experiment = experiment
        self.tracking_uri = uri
        self.last_run_id: Optional[str] = None
        logger.info(f"MLflow tracking URI: {uri} | experiment: {experiment}")

    def log_run(
        self,
        model: Any,
        params: Dict,
        metrics: Dict,
        threshold_data: Optional[Dict] = None,
        best_optuna_value: Optional[float] = None,
        artefact_paths: Optional[List[str]] = None,
        run_name: str = "lgbm-optuna-champion",
        model_name: str = "lightgbm",
    ) -> str:
        """
        Log a full training run to MLflow.
        Matches notebook section 13 log structure.
        """
        with mlflow.start_run(run_name=run_name):
            # Hyperparameters
            mlflow.log_params(params)
            if threshold_data:
                mlflow.log_param("threshold", threshold_data.get("threshold"))
                mlflow.log_param("recall_at_threshold", threshold_data.get("recall_at_threshold"))

            # Metrics
            for key in ["auc_pr", "auc_roc", "f1", "recall", "precision"]:
                if key in metrics:
                    mlflow.log_metric(key, metrics[key])
            if best_optuna_value is not None:
                mlflow.log_metric("optuna_best_aucpr", best_optuna_value)

            # Model artefact
            try:
                if model_name == "lightgbm":
                    mlflow.lightgbm.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
            except Exception as exc:
                logger.warning(f"Could not log model artefact: {exc}")

            # PNG artefacts
            for path in (artefact_paths or []):
                if os.path.exists(path):
                    mlflow.log_artifact(path)

            self.last_run_id = mlflow.active_run().info.run_id

        logger.info(f"MLflow run logged: {self.last_run_id}")
        return self.last_run_id

    def save_threshold(
        self,
        threshold: float,
        recall: float,
        precision: float,
        metrics: Dict,
        path: str = "models/threshold.json",
    ) -> None:
        """Save threshold data to JSON (notebook section 12)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "threshold": threshold,
            "recall_at_threshold": recall,
            "precision_at_threshold": precision,
            "auc_pr": metrics.get("auc_pr"),
            "f1": metrics.get("f1"),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Threshold config saved to {path}")
        logger.info(json.dumps(data, indent=2))
