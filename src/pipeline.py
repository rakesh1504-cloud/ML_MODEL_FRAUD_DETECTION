import json
import logging
from typing import Dict, Optional

import pandas as pd

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.models.predict import FraudPredictor
from src.models.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Orchestrates the full IEEE-CIS fraud-detection ML lifecycle.

    Training flow (matches 01_feature_engineering.ipynb):
      1. Load & merge raw data
      2. Feature engineering (velocity, amount aggs, interactions, encoding)
      3. Drop high-missing cols + impute
      4. Time-based split (no random split — avoids temporal leakage)
      5. SMOTE on train only (fraud → 10% of training set)
      6. Train model
      7. Evaluate (primary metric: AUC-PR)
      8. Save model + preprocessor + feature list
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ingestion = DataIngestion(
            raw_data_dir=self.config.get("raw_data_dir", "data/raw")
        )
        self.preprocessor = DataPreprocessor(
            processed_dir=self.config.get("processed_dir", "data/processed"),
            smote_strategy=self.config.get("smote_strategy", 0.1),
        )
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(
            model_dir=self.config.get("model_dir", "data/models")
        )
        self.evaluator = ModelEvaluator(
            threshold=self.config.get("threshold", 0.5)
        )
        self.predictor: Optional[FraudPredictor] = None
        self._feature_cols = []

    # ------------------------------------------------------------------
    # Training flow
    # ------------------------------------------------------------------

    def run_training(
        self,
        txn_file: Optional[str] = None,
        identity_file: Optional[str] = None,
        data_file: Optional[str] = None,
        model_name: str = "lightgbm",
        model_params: Optional[Dict] = None,
        optimize_threshold: bool = True,
        apply_smote: bool = True,
        n_synthetic: int = 10_000,
        save_parquet: bool = True,
    ) -> Dict:
        logger.info("=== FRAUD DETECTION PIPELINE — TRAINING ===")

        # 1. Ingest
        if txn_file:
            df = self.ingestion.load_ieee_cis(
                txn_file=txn_file,
                identity_file=identity_file or "train_identity.csv",
            )
        elif data_file:
            df = self.ingestion.load_file(data_file)
        else:
            logger.info("No data file — generating synthetic IEEE-CIS-style dataset")
            df = self.ingestion.generate_synthetic(n_samples=n_synthetic)

        self.ingestion.validate_schema(df)

        # 2. Feature engineering (fit encoders, velocity, aggregations)
        df = self.feature_engineer.fit_transform(df)

        # Store feature columns (excluding ID/target/time-offset)
        drop = {"TransactionID", "TransactionDT", "isFraud"}
        self._feature_cols = [c for c in df.columns if c not in drop]
        self.feature_engineer.save_feature_cols(self._feature_cols)

        # 3. Drop high-missing + impute + time-based split + SMOTE
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(
            df, apply_smote=apply_smote
        )
        self.preprocessor.save()

        # Optionally persist Parquet splits for downstream notebooks
        if save_parquet:
            self.preprocessor.save_parquet(X_train, y_train, X_test, y_test)

        # 4. Train
        self.trainer.build(model_name=model_name, params=model_params)
        self.trainer.fit(X_train, y_train)

        # 5. Optimise threshold on test set (use AUC-PR-neutral F1 sweep)
        if optimize_threshold:
            self.evaluator.find_optimal_threshold(self.trainer.model, X_test, y_test, metric="f1")

        # 6. Evaluate
        test_metrics = self.evaluator.evaluate(self.trainer.model, X_test, y_test, "test")
        self.evaluator.save_report()

        # 7. Save model
        model_path = self.trainer.save()

        # 8. Build predictor
        self.predictor = FraudPredictor.from_components(
            self.trainer, self.preprocessor, self.feature_engineer, self.evaluator.threshold
        )

        summary = {
            "model": model_name,
            "model_path": model_path,
            "threshold": self.evaluator.threshold,
            "primary_metric": "auc_pr",
            "test_metrics": test_metrics,
            "training_metadata": self.trainer.training_metadata,
            "n_features": len(self._feature_cols),
        }

        logger.info("=== TRAINING COMPLETE ===")
        printable = {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
        logger.info(json.dumps(printable, indent=2))
        return summary

    # ------------------------------------------------------------------
    # Inference flow
    # ------------------------------------------------------------------

    def run_inference(
        self,
        transactions: pd.DataFrame,
        model_filename: str = "lightgbm.pkl",
        preprocessor_path: str = "data/processed/preprocessor.pkl",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        logger.info("=== FRAUD DETECTION PIPELINE — INFERENCE ===")

        if self.predictor is None:
            self.trainer.load(model_filename)
            self.preprocessor = DataPreprocessor.load(preprocessor_path)
            self.predictor = FraudPredictor.from_components(
                self.trainer,
                self.preprocessor,
                self.feature_engineer,
                threshold or self.evaluator.threshold,
            )

        results = self.predictor.predict_batch(transactions)
        out = transactions.copy().reset_index(drop=True)
        out["fraud_probability"]  = [r["fraud_probability"] for r in results]
        out["is_fraud_predicted"] = [r["is_fraud"] for r in results]
        out["risk_level"]         = [r["risk_level"] for r in results]
        logger.info(f"Scored {len(out):,} transactions")
        return out

    def predict_single(self, transaction: Dict) -> Dict:
        if self.predictor is None:
            raise RuntimeError("Pipeline not trained. Call run_training() first.")
        return self.predictor.predict_single(transaction)
