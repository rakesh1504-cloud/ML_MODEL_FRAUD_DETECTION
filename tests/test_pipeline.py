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
    def test_training_returns_summary(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="logistic_regression",
            optimize_threshold=False,
            n_synthetic=500,
        )
        assert "test_metrics" in summary
        assert "auc_pr" in summary["test_metrics"]
        # AUC-PR must exceed the naive baseline (= fraud rate ~3.5%)
        assert summary["test_metrics"]["auc_pr"] > 0.035
        assert summary["primary_metric"] == "auc_pr"

    def test_predict_single_output_keys(self, trained_pipeline):
        # Minimal IEEE-CIS transaction
        transaction = {
            "TransactionID": 2987001,
            "TransactionDT": 7200,            # 2 hours after ref date
            "TransactionAmt": 1500.0,
            "ProductCD": "C",
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
            "card4": "mastercard",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 1.0,
        }
        result = trained_pipeline.predict_single(transaction)
        assert result["risk_level"] in {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}

    def test_transaction_amt_column_name(self, trained_pipeline):
        """Ensure old 'amount' field is not expected — only TransactionAmt."""
        transaction = {
            "TransactionAmt": 200.0,
            "TransactionDT": 3600,
            "ProductCD": "H",
        }
        # Should not raise KeyError on TransactionAmt
        result = trained_pipeline.predict_single(transaction)
        assert isinstance(result["fraud_probability"], float)
