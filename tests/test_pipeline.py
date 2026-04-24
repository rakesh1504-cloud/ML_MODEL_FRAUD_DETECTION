import pytest

from src.pipeline import FraudDetectionPipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    pipeline = FraudDetectionPipeline()
    pipeline.run_training(
        model_name="random_forest",
        model_params={"n_estimators": 20},
        optimize_threshold=False,
        apply_smote=True,
        n_synthetic=800,
        save_parquet=False,
    )
    return pipeline


class TestFraudDetectionPipeline:
    def test_training_returns_summary(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="logistic_regression",
            optimize_threshold=False,
            apply_smote=True,
            n_synthetic=500,
            save_parquet=False,
        )
        assert "test_metrics" in summary
        assert "auc_pr" in summary["test_metrics"]
        assert summary["primary_metric"] == "auc_pr"
        assert summary["test_metrics"]["auc_pr"] > 0.035  # must beat naive baseline

    def test_predict_single_output_keys(self, trained_pipeline):
        transaction = {
            "TransactionID": 2987001,
            "TransactionDT": 7200,
            "TransactionAmt": 1500.0,
            "ProductCD": "C",
            "card1": 1234,
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "anonymous.com",
            "R_emaildomain": "gmail.com",
            "C1": 12.0,
            "DeviceType": None,
        }
        result = trained_pipeline.predict_single(transaction)
        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_risk_level_valid(self, trained_pipeline):
        transaction = {
            "TransactionAmt": 30.0,
            "TransactionDT": 50000,
            "ProductCD": "W",
            "card1": 5678,
            "card4": "mastercard",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 1.0,
        }
        result = trained_pipeline.predict_single(transaction)
        assert result["risk_level"] in {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}

    def test_n_features_reported(self, trained_pipeline):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="random_forest",
            model_params={"n_estimators": 10},
            optimize_threshold=False,
            apply_smote=False,
            n_synthetic=400,
            save_parquet=False,
        )
        assert summary["n_features"] > 0
