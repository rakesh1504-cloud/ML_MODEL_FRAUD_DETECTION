import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# EDA finding: AUC-PR is the primary metric — accuracy is misleading at 3.5% fraud rate
PRIMARY_METRIC = "auc_pr"


class ModelEvaluator:
    """
    Compute and persist evaluation metrics for fraud classifiers.

    Threshold strategy (notebook section 7):
      Banking context — missing fraud (FN) costs more than a false alarm (FP).
      Default: find the lowest threshold where recall >= recall_target (90%).
      F1-sweep is also available when precision/recall trade-off is balanced.
    """

    def __init__(self, threshold: float = 0.5, recall_target: float = 0.90):
        self.threshold = threshold
        self.recall_target = recall_target
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
            # Primary metric (notebook section 2 helper)
            "auc_pr": round(average_precision_score(y, y_proba), 4),
            "auc_roc": round(roc_auc_score(y, y_proba), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
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
        metric: str = "recall_target",
    ) -> float:
        """
        Two strategies:
          'recall_target' — lowest threshold where recall >= self.recall_target (default).
                            Banking context: catching fraud matters more than false alarms.
          'f1'            — threshold that maximises F1 on the validation set.
        """
        y_proba = model.predict_proba(X_val)[:, 1]

        if metric == "recall_target":
            thresh, prec, rec = self.tune_threshold_by_recall(y_val, y_proba, self.recall_target)
            logger.info(
                f"Threshold at recall>={self.recall_target:.0%}: "
                f"{thresh:.3f}  (recall={rec:.4f}, precision={prec:.4f})"
            )
            self.threshold = thresh
            return thresh

        # F1 sweep
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thresh, best_score = self.threshold, -1.0
        for t in thresholds:
            score = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
            if score > best_score:
                best_score, best_thresh = score, t
        logger.info(f"Optimal F1 threshold: {best_thresh:.2f} (F1={best_score:.4f})")
        self.threshold = best_thresh
        return best_thresh

    @staticmethod
    def tune_threshold_by_recall(
        y_true: pd.Series,
        y_prob: np.ndarray,
        recall_target: float = 0.90,
    ) -> Tuple[float, float, float]:
        """
        Find the lowest threshold where recall >= recall_target.
        Returns (threshold, precision, recall) at that point.
        Matches notebook section 2 `tune_threshold` helper.
        """
        prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_prob)
        for p, r, t in zip(prec_arr, rec_arr, thresholds):
            if r >= recall_target:
                return float(t), float(p), float(r)
        return 0.5, float(prec_arr[-1]), float(rec_arr[-1])

    def business_impact(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        avg_fraud_amt: float = 250.0,
        avg_legit_amt: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Translate model performance into business dollar impact.
        Matches notebook section 11 framing for banking stakeholders.
        """
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fraud_blocked_usd = tp * avg_fraud_amt
        fraud_missed_usd  = fn * avg_fraud_amt
        false_alarm_usd   = fp * avg_legit_amt
        total_fraud_usd   = (tp + fn) * avg_fraud_amt
        pct_blocked       = fraud_blocked_usd / max(total_fraud_usd, 1) * 100

        impact = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "fraud_blocked_usd": round(fraud_blocked_usd, 2),
            "fraud_missed_usd":  round(fraud_missed_usd, 2),
            "false_alarm_usd":   round(false_alarm_usd, 2),
            "pct_fraud_blocked": round(pct_blocked, 2),
            "fraud_catch_rate":  round(tp / max(tp + fn, 1) * 100, 2),
            "false_alarm_rate":  round(fp / max(fp + tn, 1) * 100, 2),
        }

        logger.info("=== BUSINESS IMPACT ===")
        logger.info(f"  Fraud blocked : {tp:,} txns = ${fraud_blocked_usd:,.0f}")
        logger.info(f"  Fraud missed  : {fn:,} txns = ${fraud_missed_usd:,.0f}")
        logger.info(f"  False alarms  : {fp:,} txns = ${false_alarm_usd:,.0f}")
        logger.info(f"  % Fraud blocked: {pct_blocked:.1f}%")
        return impact

    def save_report(self, path: str = "data/models/evaluation_report.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {path}")

    def _log(self, m: Dict) -> None:
        cm = m["confusion_matrix"]
        logger.info(
            f"[{m['dataset']}] AUC-PR={m['auc_pr']:.4f} (primary)  "
            f"AUC-ROC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
            f"Recall={m['recall']:.4f}  Precision={m['precision']:.4f}"
        )
        logger.info(
            f"  CM — TN:{cm['tn']:,} FP:{cm['fp']:,} FN:{cm['fn']:,} TP:{cm['tp']:,}"
        )
