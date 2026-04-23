import pytest

from src.pipeline import FraudDetectionPipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    pipeline = FraudDetectionPipeline()
    pipeline.run_training(
        model_name="random_forest",
        model_params={"n_estimators": 20},
        optimize_threshold=False,
        n_synthetic=800,
    )
    return pipeline


class TestFraudDetectionPipeline:
    def test_training_returns_summary(self, trained_pipeline):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="logistic_regression",
            optimize_threshold=False,
            n_synthetic=500,
        )
        assert "test_metrics" in summary
        assert summary["test_metrics"]["roc_auc"] > 0.5

    def test_predict_single_output_keys(self, trained_pipeline):
        transaction = {
            "transaction_id": "TXN00000001",
            "amount": 1500.0,
            "hour": 3,
            "day_of_week": 6,
            "merchant_category": "online",
            "card_present": 0,
            "distance_from_home_km": 900.0,
            "num_transactions_last_24h": 12,
            "is_foreign_transaction": 1,
        }
        result = trained_pipeline.predict_single(transaction)
        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_risk_level_valid(self, trained_pipeline):
        transaction = {
            "amount": 50.0,
            "hour": 10,
            "day_of_week": 2,
            "merchant_category": "grocery",
            "card_present": 1,
            "distance_from_home_km": 2.0,
            "num_transactions_last_24h": 1,
            "is_foreign_transaction": 0,
        }
        result = trained_pipeline.predict_single(transaction)
        assert result["risk_level"] in {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}
