import logging
from typing import Dict, List

import pandas as pd

from src.data.preprocessing import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer

logger = logging.getLogger(__name__)


class FraudPredictor:
    """End-to-end inference: raw IEEE-CIS transaction → fraud score + label."""

    def __init__(
        self,
        model_filename: str = "lightgbm.pkl",
        preprocessor_path: str = "data/processed/preprocessor.pkl",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self._trainer = ModelTrainer()
        self._trainer.load(model_filename)
        self._preprocessor = DataPreprocessor.load(preprocessor_path)
        self._feature_engineer = FeatureEngineer()

    @classmethod
    def from_components(
        cls,
        trainer: ModelTrainer,
        preprocessor: DataPreprocessor,
        feature_engineer: FeatureEngineer,
        threshold: float = 0.5,
    ) -> "FraudPredictor":
        obj = cls.__new__(cls)
        obj.threshold = threshold
        obj._trainer = trainer
        obj._preprocessor = preprocessor
        obj._feature_engineer = feature_engineer
        return obj

    def predict_single(self, transaction: Dict) -> Dict:
        return self.predict_batch(pd.DataFrame([transaction]))[0]

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        df_eng = self._feature_engineer.transform(df)
        # Fill any columns expected by the preprocessor that are absent in this input.
        # At training time those columns existed; at inference they may not be provided.
        for col in getattr(self._preprocessor, "numeric_cols", []):
            if col not in df_eng.columns:
                df_eng[col] = float("nan")
        X = self._preprocessor.transform(df_eng)
        proba = self._trainer.model.predict_proba(X)[:, 1]
        labels = (proba >= self.threshold).astype(int)
        return [
            {
                "fraud_probability": round(float(p), 4),
                "is_fraud": bool(l),
                "risk_level": self._risk_level(float(p)),
            }
            for p, l in zip(proba, labels)
        ]

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.8:
            return "HIGH"
        if score >= 0.5:
            return "MEDIUM"
        if score >= 0.2:
            return "LOW"
        return "VERY_LOW"
