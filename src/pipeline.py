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

    Data source options (in order of priority):
      1. Real IEEE-CIS data  : provide txn_file + identity_file
      2. Single CSV/Parquet  : provide data_file
      3. Synthetic data      : default when no file is given
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ingestion = DataIngestion(
            raw_data_dir=self.config.get("raw_data_dir", "data/raw")
        )
        self.preprocessor = DataPreprocessor(
            processed_dir=self.config.get("processed_dir", "data/processed")
        )
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(
            model_dir=self.config.get("model_dir", "data/models")
        )
        self.evaluator = ModelEvaluator(
            threshold=self.config.get("threshold", 0.5)
        )
        self.predictor: Optional[FraudPredictor] = None

    # ------------------------------------------------------------------
    # Training flow
    # ------------------------------------------------------------------

    def run_training(
        self,
        txn_file: Optional[str] = None,
        identity_file: Optional[str] = None,
        data_file: Optional[str] = None,
        model_name: str = "random_forest",
        model_params: Optional[Dict] = None,
        optimize_threshold: bool = True,
        n_synthetic: int = 10_000,
    ) -> Dict:
        logger.info("=== FRAUD DETECTION PIPELINE — TRAINING ===")

        # 1. Ingest data
        if txn_file:
            # IEEE-CIS: merge transaction + identity tables
            df = self.ingestion.load_ieee_cis(
                txn_file=txn_file,
                identity_file=identity_file or "train_identity.csv",
            )
        elif data_file:
            df = self.ingestion.load_file(data_file)
        else:
            logger.info("No data file provided — generating synthetic IEEE-CIS-style dataset")
            df = self.ingestion.generate_synthetic(n_samples=n_synthetic)

        self.ingestion.validate_schema(df)

        # 2. Feature engineering (before preprocessing so raw values are intact)
        df = self.feature_engineer.transform(df)

        # 3. Auto-compute scale_pos_weight from class distribution (EDA finding)
        if model_params is None:
            model_params = {}
        if "scale_pos_weight" not in model_params and model_name == "gradient_boosting":
            fraud_count = int(df["isFraud"].sum())
            legit_count = len(df) - fraud_count
            if fraud_count > 0:
                spw = round(legit_count / fraud_count, 1)
                model_params["scale_pos_weight"] = spw
                logger.info(f"Auto scale_pos_weight = {spw} (legit/fraud = {legit_count}/{fraud_count})")

        # 4. Preprocess + split
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.fit_transform(df)
        self.preprocessor.save()

        # 5. Train
        self.trainer.build(model_name=model_name, params=model_params)
        self.trainer.fit(X_train, y_train, X_val, y_val)

        # 6. Optimise decision threshold on validation set
        if optimize_threshold:
            self.evaluator.find_optimal_threshold(self.trainer.model, X_val, y_val, metric="f1")

        # 7. Evaluate on val + test
        self.evaluator.evaluate(self.trainer.model, X_val, y_val, "validation")
        test_metrics = self.evaluator.evaluate(self.trainer.model, X_test, y_test, "test")
        self.evaluator.save_report()

        # 8. Persist model
        model_path = self.trainer.save()

        # 9. Build predictor for immediate inference
        self.predictor = FraudPredictor.from_components(
            self.trainer, self.preprocessor, self.evaluator.threshold
        )

        summary = {
            "model": model_name,
            "model_path": model_path,
            "threshold": self.evaluator.threshold,
            "primary_metric": "auc_pr",
            "test_metrics": test_metrics,
            "training_metadata": self.trainer.training_metadata,
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
        model_filename: str = "random_forest.pkl",
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
                threshold or self.evaluator.threshold,
            )

        results = self.predictor.predict_batch(transactions)
        out = transactions.copy().reset_index(drop=True)
        out["fraud_probability"] = [r["fraud_probability"] for r in results]
        out["is_fraud_predicted"] = [r["is_fraud"] for r in results]
        out["risk_level"] = [r["risk_level"] for r in results]
        logger.info(f"Scored {len(out):,} transactions")
        return out

    def predict_single(self, transaction: Dict) -> Dict:
        if self.predictor is None:
            raise RuntimeError("Pipeline not trained. Call run_training() first.")
        return self.predictor.predict_single(transaction)
