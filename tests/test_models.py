import pytest
import pandas as pd

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer, SUPPORTED_MODELS
from src.models.evaluate import ModelEvaluator, PRIMARY_METRIC


@pytest.fixture(scope="module")
def prepared_data():
    df = DataIngestion(raw_data_dir="data/raw").generate_synthetic(
        n_samples=1000, fraud_rate=0.05, save=False
    )
    df = FeatureEngineer().fit_transform(df)
    pre = DataPreprocessor()
    X_train, X_test, y_train, y_test = pre.fit_transform(df, apply_smote=True)
    return X_train, X_test, y_train, y_test


class TestModelTrainer:
    def test_build_all_model_types(self):
        trainer = ModelTrainer(model_dir="data/models")
        for name in SUPPORTED_MODELS:
            assert trainer.build(model_name=name) is not None

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelTrainer().build(model_name="does_not_exist")

    def test_fit_reports_auc_pr(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_train, y_train, X_test, y_test)
        assert "val_auc_pr" in trainer.training_metadata
        assert 0 < trainer.training_metadata["val_auc_pr"] <= 1

    def test_scale_pos_weight_extracted(self):
        trainer = ModelTrainer()
        trainer.build("gradient_boosting", params={"n_estimators": 5, "scale_pos_weight": 9})
        assert trainer._scale_pos_weight == 9

    def test_feature_importances(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_train, y_train)
        imp = trainer.feature_importances(list(X_train.columns))
        assert len(imp) == X_train.shape[1]

    def test_save_and_load(self, prepared_data, tmp_path):
        X_train, X_test, y_train, y_test = prepared_data
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.build("logistic_regression")
        trainer.fit(X_train, y_train)
        trainer.save("test_model.pkl")
        loader = ModelTrainer(model_dir=str(tmp_path))
        loader.load("test_model.pkl")
        assert len(loader.model.predict(X_test)) == len(X_test)


class TestModelEvaluator:
    def test_auc_pr_is_primary_metric(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_train, y_train)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X_test, y_test)
        assert PRIMARY_METRIC in metrics
        assert "accuracy" in metrics  # reported but not primary

    def test_auc_pr_beats_naive_baseline(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        trainer = ModelTrainer(model_dir="data/models")
        trainer.build("random_forest", params={"n_estimators": 10})
        trainer.fit(X_train, y_train)
        metrics = ModelEvaluator().evaluate(trainer.model, X_test, y_test)
        assert metrics["auc_pr"] > float(y_test.mean())
