import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute and persist evaluation metrics for fraud classifiers."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.metrics: Dict[str, Any] = {}

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_label: str = "test",
    ) -> Dict[str, Any]:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "dataset": dataset_label,
            "threshold": self.threshold,
            "n_samples": len(y),
            "fraud_rate": float(y.mean()),
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
            "average_precision": round(average_precision_score(y, y_proba), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        self.metrics[dataset_label] = metrics
        self._log_metrics(metrics)
        return metrics

    def find_optimal_threshold(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "f1",
    ) -> float:
        y_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        best_thresh, best_score = self.threshold, -1.0
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_val, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            if score > best_score:
                best_score, best_thresh = score, t
        logger.info(f"Optimal threshold ({metric}={best_score:.4f}): {best_thresh:.2f}")
        self.threshold = best_thresh
        return best_thresh

    def save_report(self, path: str = "data/models/evaluation_report.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {path}")

    def _log_metrics(self, m: Dict) -> None:
        logger.info(
            f"[{m['dataset']}] acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} "
            f"avg_prec={m['average_precision']:.4f}"
        )
        cm = m["confusion_matrix"]
        logger.info(f"  Confusion matrix — TN:{cm['tn']} FP:{cm['fp']} FN:{cm['fn']} TP:{cm['tp']}")
