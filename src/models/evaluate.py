import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# EDA finding (section 3): AUC-PR is the primary metric for imbalanced fraud data.
# Accuracy is misleading — a model predicting "not fraud" always scores ~96.5%.
PRIMARY_METRIC = "auc_pr"


class ModelEvaluator:
    """
    Compute and persist evaluation metrics for fraud classifiers.

    Primary metric: AUC-PR (average precision score).
    Secondary metric: ROC-AUC.
    Accuracy is reported but NOT used for model selection.
    """

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
            "fraud_rate": round(float(y.mean()), 4),
            # Primary metric (EDA section 3)
            "auc_pr": round(average_precision_score(y, y_proba), 4),
            # Secondary metric
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
            # Threshold-dependent metrics
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            # Reported only — not used for selection (EDA: misleading on imbalanced data)
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        self.metrics[dataset_label] = metrics
        self._log(metrics)
        return metrics

    def find_optimal_threshold(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "f1",
    ) -> float:
        """Sweep thresholds on validation set and pick the one maximising `metric`."""
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
            elif metric == "auc_pr":
                score = average_precision_score(y_val, y_proba)
                # AUC-PR doesn't depend on threshold — return immediately
                logger.info(f"AUC-PR (threshold-independent): {score:.4f}")
                return self.threshold
            else:
                raise ValueError(f"Unknown metric: {metric}")
            if score > best_score:
                best_score, best_thresh = score, t

        self.threshold = best_thresh
        logger.info(f"Optimal threshold ({metric}={best_score:.4f}): {best_thresh:.2f}")
        return best_thresh

    def save_report(self, path: str = "data/models/evaluation_report.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {path}")

    def _log(self, m: Dict) -> None:
        cm = m["confusion_matrix"]
        logger.info(
            f"[{m['dataset']}] "
            f"AUC-PR={m['auc_pr']:.4f} (primary)  "
            f"ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"Prec={m['precision']:.4f}  "
            f"Rec={m['recall']:.4f}  "
            f"Acc={m['accuracy']:.4f} (not for selection)"
        )
        logger.info(
            f"  Confusion — TN:{cm['tn']:,} FP:{cm['fp']:,} FN:{cm['fn']:,} TP:{cm['tp']:,}"
        )
