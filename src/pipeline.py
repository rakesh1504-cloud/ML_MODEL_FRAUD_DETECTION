import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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

    Training flow (based on notebooks 01 + 02):
      1.  Load & merge raw data
      2.  Feature engineering (velocity, amount aggs, interactions, encoding)
      3.  Time-based split + SMOTE on train only
      4.  Train model (LR / RF / GBM / LightGBM / XGBoost)
      5.  Optional Optuna hyperparameter tuning
      6.  Threshold tuning (recall-target strategy, default recall >= 90%)
      7.  Evaluate (primary metric: AUC-PR)
      8.  Business impact framing
      9.  Optional SHAP explainability
      10. Optional MLflow experiment tracking
      11. Save champion model + threshold.json
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ingestion = DataIngestion(raw_data_dir=self.config.get("raw_data_dir", "data/raw"))
        self.preprocessor = DataPreprocessor(
            processed_dir=self.config.get("processed_dir", "data/processed"),
            smote_strategy=self.config.get("smote_strategy", 0.1),
        )
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(model_dir=self.config.get("model_dir", "data/models"))
        self.evaluator = ModelEvaluator(
            threshold=self.config.get("threshold", 0.5),
            recall_target=self.config.get("recall_target", 0.90),
        )
        self.predictor: Optional[FraudPredictor] = None
        self._feature_cols: List[str] = []

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
        apply_smote: bool = True,
        n_synthetic: int = 10_000,
        save_parquet: bool = True,
        # Optuna tuning (notebook section 5)
        use_optuna: bool = False,
        optuna_trials: int = 50,
        # Threshold strategy (notebook section 7)
        threshold_metric: str = "recall_target",
        # SHAP explainability (notebook section 10)
        run_shap: bool = False,
        shap_sample_size: int = 2000,
        # MLflow tracking (notebook section 13)
        use_mlflow: bool = False,
        mlflow_experiment: str = "fraud-detection-v1",
        dagshub_user: Optional[str] = None,
        # Business impact (notebook section 11)
        avg_fraud_amt: float = 250.0,
        avg_legit_amt: float = 100.0,
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

        # 2. Feature engineering
        df = self.feature_engineer.fit_transform(df)
        drop = {"TransactionID", "TransactionDT", "isFraud"}
        self._feature_cols = [c for c in df.columns if c not in drop]
        self.feature_engineer.save_feature_cols(self._feature_cols)

        # 3. Preprocess: drop high-missing, impute, time-split, SMOTE
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(
            df, apply_smote=apply_smote
        )
        self.preprocessor.save()
        if save_parquet:
            self.preprocessor.save_parquet(X_train, y_train, X_test, y_test)

        # 4. Optuna tuning or standard training
        best_optuna_value: Optional[float] = None
        best_params = model_params or {}

        if use_optuna:
            from src.models.tune import OptunaHyperparamTuner
            tuner = OptunaHyperparamTuner(X_train, y_train, X_test, y_test)
            if model_name in ("lightgbm", "lgbm"):
                best_params = tuner.tune_lightgbm(n_trials=optuna_trials)
            elif model_name == "xgboost":
                best_params = tuner.tune_xgboost(n_trials=optuna_trials)
            best_optuna_value = tuner.best_value
            self.trainer.model = tuner.best_model
            self.trainer.model_name = model_name
            self.trainer.training_metadata = {
                "model_name": model_name,
                "n_train": len(X_train),
                "n_features": X_train.shape[1],
                "optuna_trials": optuna_trials,
                "optuna_best_auc_pr": round(best_optuna_value, 4),
            }
        else:
            self.trainer.build(model_name=model_name, params=best_params)
            self.trainer.fit(X_train, y_train, X_test, y_test)

        # 5. Threshold tuning (recall-target strategy from notebook section 7)
        self.evaluator.find_optimal_threshold(
            self.trainer.model, X_test, y_test, metric=threshold_metric
        )

        # 6. Evaluate
        test_metrics = self.evaluator.evaluate(self.trainer.model, X_test, y_test, "test")
        self.evaluator.save_report()

        # 7. Business impact
        impact = self.evaluator.business_impact(
            self.trainer.model, X_test, y_test, avg_fraud_amt, avg_legit_amt
        )

        # 8. SHAP explainability
        shap_top_features: List = []
        if run_shap:
            try:
                from src.models.explain import SHAPExplainer
                explainer = SHAPExplainer(
                    self.trainer.model, self._feature_cols,
                    sample_size=shap_sample_size,
                )
                explainer.compute(X_test)
                explainer.plot_beeswarm()
                explainer.plot_bar()
                y_proba = self.trainer.model.predict_proba(X_test)[:, 1]
                explainer.plot_waterfall(X_test, y_test, y_proba)
                shap_top_features = explainer.top_features(10).to_dict("records")
            except Exception as exc:
                logger.warning(f"SHAP explainability failed: {exc}")

        # 9. Save champion model + threshold.json
        model_path = self.trainer.save()
        self.trainer.save_champion(f"models/{model_name}_champion.pkl")

        pr_thresh, pr_prec, pr_rec = self.evaluator.tune_threshold_by_recall(
            y_test,
            self.trainer.model.predict_proba(X_test)[:, 1],
            self.evaluator.recall_target,
        )
        threshold_data = {
            "threshold": self.evaluator.threshold,
            "recall_at_threshold": pr_rec,
            "precision_at_threshold": pr_prec,
            "auc_pr": test_metrics["auc_pr"],
            "f1": test_metrics["f1"],
        }
        Path("models").mkdir(exist_ok=True)
        with open("models/threshold.json", "w") as f:
            json.dump(threshold_data, f, indent=2)

        # 10. MLflow tracking
        if use_mlflow:
            try:
                from src.models.tracking import MLflowTracker
                tracker = MLflowTracker(
                    experiment=mlflow_experiment,
                    dagshub_user=dagshub_user,
                )
                artefacts = [
                    "data/external/shap_beeswarm.png",
                    "data/external/shap_bar.png",
                    "data/external/shap_waterfall.png",
                    "data/external/model_comparison_pr.png",
                    "data/external/model_threshold_tuning.png",
                ]
                tracker.log_run(
                    model=self.trainer.model,
                    params=best_params,
                    metrics=test_metrics,
                    threshold_data=threshold_data,
                    best_optuna_value=best_optuna_value,
                    artefact_paths=artefacts,
                    model_name=model_name,
                )
            except Exception as exc:
                logger.warning(f"MLflow logging failed: {exc}")

        # 11. Build predictor
        self.predictor = FraudPredictor.from_components(
            self.trainer, self.preprocessor, self.feature_engineer,
            self.evaluator.threshold,
        )

        summary = {
            "model": model_name,
            "model_path": model_path,
            "champion_path": f"models/{model_name}_champion.pkl",
            "threshold": self.evaluator.threshold,
            "primary_metric": "auc_pr",
            "test_metrics": test_metrics,
            "business_impact": impact,
            "training_metadata": self.trainer.training_metadata,
            "n_features": len(self._feature_cols),
            "shap_top_features": shap_top_features,
        }

        logger.info("=== TRAINING COMPLETE ===")
        printable = {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
        logger.info(json.dumps(printable, indent=2))
        return summary

    # ------------------------------------------------------------------
    # Inference
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
                self.trainer, self.preprocessor, self.feature_engineer,
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
