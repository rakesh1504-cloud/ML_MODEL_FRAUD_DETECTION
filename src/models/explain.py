import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based model explainability.

    Three plots (notebook section 10):
      - beeswarm  : global view of all features across predictions
      - bar       : mean |SHAP| feature importance
      - waterfall : single transaction explanation

    Works with tree-based models (LightGBM, XGBoost, RandomForest).
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        output_dir: str = "data/external",
        sample_size: int = 2000,
        random_state: int = 42,
    ):
        if not _SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap")
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.random_state = random_state
        self._explainer: Optional[Any] = None
        self._shap_values: Optional[np.ndarray] = None
        self._X_sample: Optional[pd.DataFrame] = None

    def compute(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values on a random sample of X."""
        rng = np.random.default_rng(self.random_state)
        n = min(self.sample_size, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        self._X_sample = X.iloc[idx].reset_index(drop=True)

        logger.info(f"Computing SHAP values on {n:,} samples …")
        self._explainer = shap.TreeExplainer(self.model)
        sv = self._explainer.shap_values(self._X_sample)

        # LightGBM returns [class0, class1] — take class1 (fraud)
        self._shap_values = sv[1] if isinstance(sv, list) else sv
        logger.info(f"SHAP values shape: {self._shap_values.shape}")
        return self._shap_values

    def plot_beeswarm(self, max_display: int = 20, save: bool = True) -> None:
        """Global beeswarm: each dot = one transaction (notebook section 10)."""
        self._check_computed()
        plt.figure(figsize=(12, 9))
        shap.summary_plot(
            self._shap_values, self._X_sample,
            feature_names=self.feature_names,
            max_display=max_display, show=False,
        )
        plt.title("SHAP Beeswarm — Top features driving fraud score",
                  fontweight="bold", pad=12)
        plt.tight_layout()
        if save:
            path = self.output_dir / "shap_beeswarm.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved {path}")
        plt.show()
        plt.close()

    def plot_bar(self, max_display: int = 20, save: bool = True) -> None:
        """Mean |SHAP| bar chart — aggregate feature importance."""
        self._check_computed()
        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            self._shap_values, self._X_sample,
            feature_names=self.feature_names,
            plot_type="bar", max_display=max_display, show=False,
        )
        plt.title("SHAP Feature Importance (mean |SHAP value|)",
                  fontweight="bold", pad=12)
        plt.tight_layout()
        if save:
            path = self.output_dir / "shap_bar.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved {path}")
        plt.show()
        plt.close()

    def plot_waterfall(
        self,
        X_full: pd.DataFrame,
        y_true: pd.Series,
        y_proba: np.ndarray,
        max_display: int = 15,
        save: bool = True,
    ) -> None:
        """
        Single-transaction waterfall: why was THIS transaction flagged?
        Picks the correctly-identified fraud with the highest predicted probability.
        Matches notebook section 10.
        """
        self._check_computed()
        fraud_idx = np.where(y_true.values == 1)[0]
        if len(fraud_idx) == 0:
            logger.warning("No fraud transactions found for waterfall plot")
            return

        best_pos = fraud_idx[np.argmax(y_proba[fraud_idx])]
        single = self._explainer(X_full.iloc[[best_pos]])

        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(single[0], max_display=max_display, show=False)
        plt.title(
            f"SHAP Waterfall — Why transaction #{best_pos} was flagged as fraud",
            fontweight="bold",
        )
        plt.tight_layout()
        if save:
            path = self.output_dir / "shap_waterfall.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved {path}")
        plt.show()
        plt.close()

    def top_features(self, n: int = 10) -> pd.DataFrame:
        """Return top-n features by mean |SHAP| value."""
        self._check_computed()
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def _check_computed(self) -> None:
        if self._shap_values is None:
            raise RuntimeError("Call compute(X) before plotting.")
