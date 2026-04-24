import pytest
import pandas as pd
import numpy as np

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer, SUPPORTED_MODELS
from src.models.evaluate import ModelEvaluator, PRIMARY_METRIC


@pytest.fixture(scope="module")
def prepared_data():
    ingestion = DataIngestion(raw_data_dir="data/raw")
    df = ingestion.generate_synthetic(n_samples=1000, fraud_rate=0.05, save=False)
    df = FeatureEngineer().transform(df)
    preprocessor = DataPreprocessor()
    X_tr, X_val, X_te, y_tr, y_val, y_te = preprocessor.fit_transform(df)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


class TestModelTrainer:
    def test_build_all_model_types(self):
        trainer = ModelTrainer(model_dir="data/models")
        for name in SUPPORTED_MODELS:
            model = trainer.build(model_name=name)
            assert model is not None

    def test_build_unknown_raises(self):
        trainer = ModelTrainer(model_dir="data/models")
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.build(model_name="does_not_exist")

    def test_fit_random_forest(self, prepared_data):
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_tr, y_tr, X_val, y_val)
        assert "val_auc_pr" in trainer.training_metadata
        assert 0 < trainer.training_metadata["val_auc_pr"] <= 1

    def test_scale_pos_weight_extracted(self, prepared_data):
        """scale_pos_weight should be stored but not passed to sklearn."""
        X_tr, *_ = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("gradient_boosting", params={"n_estimators": 5, "scale_pos_weight": 27})
        assert trainer._scale_pos_weight == 27

    def test_feature_importances(self, prepared_data):
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_tr, y_tr)
        imp = trainer.feature_importances(list(X_tr.columns))
        assert len(imp) == X_tr.shape[1]
        assert imp["importance"].sum() == pytest.approx(1.0, abs=1e-3)

    def test_save_and_load(self, prepared_data, tmp_path):
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.build("logistic_regression")
        trainer.fit(X_tr, y_tr)
        trainer.save("test_model.pkl")

        loader = ModelTrainer(model_dir=str(tmp_path))
        loader.load("test_model.pkl")
        preds = loader.model.predict(X_te)
        assert len(preds) == len(X_te)


class TestModelEvaluator:
    def test_evaluate_returns_primary_metric(self, prepared_data):
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_tr, y_tr)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X_te, y_te)
        # AUC-PR is the primary metric (EDA section 3)
        assert PRIMARY_METRIC in metrics
        assert "roc_auc" in metrics
        assert "accuracy" in metrics  # reported but not primary

    def test_auc_pr_beats_naive_baseline(self, prepared_data):
        """AUC-PR should exceed the naive baseline (= fraud rate)."""
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_tr, y_tr)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X_te, y_te)
        naive_baseline = float(y_te.mean())
        assert metrics["auc_pr"] > naive_baseline

    def test_optimal_threshold_in_range(self, prepared_data):
        X_tr, X_val, X_te, y_tr, y_val, y_te = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_tr, y_tr)
        evaluator = ModelEvaluator()
        thresh = evaluator.find_optimal_threshold(trainer.model, X_val, y_val)
        assert 0.1 <= thresh <= 0.9
