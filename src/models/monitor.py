import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Run: pip install evidently")


class DriftMonitor:
    """
    Data drift monitoring using Evidently AI.

    Compares reference (training) distribution against current (production/test)
    distribution and generates an HTML report + structured drift summary.

    Usage:
        monitor = DriftMonitor(reference_data=X_train, n_cols=30)
        summary = monitor.run(current_data=X_test)
        monitor.save_report("data/external/drift_report.html")
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        n_cols: int = 30,
        output_dir: str = "data/external",
    ):
        if not _EVIDENTLY_AVAILABLE:
            raise ImportError("Install evidently: pip install evidently")

        num_cols = reference_data.select_dtypes(include="number").columns[:n_cols].tolist()
        self.reference = reference_data[num_cols].copy()
        self.num_cols  = num_cols
        self.output_dir = Path(output_dir)
        self._report: Optional[Report] = None
        self._result: Optional[Dict]   = None

    def run(
        self,
        current_data: pd.DataFrame,
        ref_sample: int = 5000,
        curr_sample: int = 2000,
        random_state: int = 42,
    ) -> Dict:
        """
        Run drift detection and return a summary dict.

        Returns keys:
            dataset_drift (bool), drifted_features (int),
            total_features (int), missing_values_share (float)
        """
        ref  = self.reference.sample(n=min(ref_sample,  len(self.reference)),  random_state=random_state)
        curr = current_data[self.num_cols].sample(n=min(curr_sample, len(current_data)), random_state=random_state)

        logger.info(f"Running drift report — reference: {ref.shape}, current: {curr.shape}")

        self._report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftPreset(),
        ])
        self._report.run(reference_data=ref, current_data=curr)
        self._result = self._report.as_dict()

        drift_metric   = self._result["metrics"][0]["result"]
        missing_metric = self._result["metrics"][1]["result"]

        summary = {
            "dataset_drift":      drift_metric.get("dataset_drift", False),
            "drifted_features":   drift_metric.get("number_of_drifted_columns", 0),
            "total_features":     drift_metric.get("number_of_columns", len(self.num_cols)),
            "missing_values_share": missing_metric.get("current", {}).get("share_of_missing_values", 0.0),
        }

        logger.info(
            f"Drift: {summary['dataset_drift']} | "
            f"Drifted features: {summary['drifted_features']}/{summary['total_features']}"
        )
        return summary

    def save_report(self, path: Optional[str] = None) -> str:
        if self._report is None:
            raise RuntimeError("Run drift detection first via .run()")
        out = path or str(self.output_dir / "drift_report.html")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        self._report.save_html(out)
        logger.info(f"Drift report saved to {out}")
        return out

    @classmethod
    def from_parquet(
        cls,
        train_path: str = "data/processed/X_train.parquet",
        n_cols: int = 30,
        output_dir: str = "data/external",
    ) -> "DriftMonitor":
        """Convenience constructor — loads reference data from saved parquet."""
        X_train = pd.read_parquet(train_path)
        return cls(reference_data=X_train, n_cols=n_cols, output_dir=output_dir)

    def run_and_save(
        self,
        current_data: pd.DataFrame,
        report_path: Optional[str] = None,
    ) -> Dict:
        """Run drift detection and save HTML report in one call."""
        summary = self.run(current_data)
        saved   = self.save_report(report_path)
        summary["report_path"] = saved
        return summary
