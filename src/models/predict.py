import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.data.preprocessing import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer

logger = logging.getLogger(__name__)


class FraudPredictor:
    """End-to-end inference: raw transaction dict → fraud score + label."""

    def __init__(
        self,
        model_path: str = "data/models/random_forest.pkl",
        preprocessor_path: str = "data/processed/preprocessor.pkl",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self._trainer = ModelTrainer()
        self._trainer.load(model_path.split("/")[-1])
        self._preprocessor = DataPreprocessor.load(preprocessor_path)
        self._feature_engineer = FeatureEngineer()

    @classmethod
    def from_components(
        cls,
        trainer: ModelTrainer,
        preprocessor: DataPreprocessor,
        threshold: float = 0.5,
    ) -> "FraudPredictor":
        obj = cls.__new__(cls)
        obj.threshold = threshold
        obj._trainer = trainer
        obj._preprocessor = preprocessor
        obj._feature_engineer = FeatureEngineer()
        return obj

    def predict_single(self, transaction: Dict) -> Dict:
        df = pd.DataFrame([transaction])
        result = self.predict_batch(df)
        return result[0]

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        df_engineered = self._feature_engineer.transform(df)
        X = self._preprocessor.transform(df_engineered)
        proba = self._trainer.model.predict_proba(X)[:, 1]
        labels = (proba >= self.threshold).astype(int)
        results = []
        for i, (score, label) in enumerate(zip(proba, labels)):
            results.append({
                "fraud_probability": round(float(score), 4),
                "is_fraud": bool(label),
                "risk_level": self._risk_level(float(score)),
            })
        return results

    def _risk_level(self, score: float) -> str:
        if score >= 0.8:
            return "HIGH"
        if score >= 0.5:
            return "MEDIUM"
        if score >= 0.2:
            return "LOW"
        return "VERY_LOW"
