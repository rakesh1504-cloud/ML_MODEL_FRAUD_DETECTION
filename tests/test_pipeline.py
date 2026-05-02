import pytest

from src.pipeline import FraudDetectionPipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    pipeline = FraudDetectionPipeline()
    pipeline.run_training(
        model_name="random_forest",
        model_params={"n_estimators": 20},
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
            apply_smote=True,
            n_synthetic=500,
            save_parquet=False,
        )
        assert "test_metrics" in summary
        assert "auc_pr" in summary["test_metrics"]
        assert summary["primary_metric"] == "auc_pr"
        assert summary["test_metrics"]["auc_pr"] > 0.035  # must beat naive baseline

    def test_summary_has_required_keys(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="logistic_regression",
            apply_smote=False,
            n_synthetic=400,
            save_parquet=False,
        )
        for key in ["model", "model_path", "threshold", "primary_metric",
                    "test_metrics", "business_impact", "training_metadata", "n_features"]:
            assert key in summary, f"Missing key: {key}"

    def test_n_features_reported(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="random_forest",
            model_params={"n_estimators": 10},
            apply_smote=False,
            n_synthetic=400,
            save_parquet=False,
        )
        assert summary["n_features"] > 0

    def test_business_impact_keys(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="random_forest",
            model_params={"n_estimators": 10},
            apply_smote=False,
            n_synthetic=400,
            save_parquet=False,
        )
        impact = summary["business_impact"]
        for key in ["fraud_blocked_usd", "fraud_missed_usd", "pct_fraud_blocked", "fraud_catch_rate"]:
            assert key in impact, f"Missing business impact key: {key}"

    def test_threshold_is_float(self):
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="logistic_regression",
            apply_smote=False,
            n_synthetic=400,
            save_parquet=False,
            threshold_metric="recall_target",
        )
        assert isinstance(summary["threshold"], float)
        assert 0.0 <= summary["threshold"] <= 1.0

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

    def test_xgboost_pipeline(self):
        """XGBoost should run end-to-end through the pipeline without errors."""
        pipeline = FraudDetectionPipeline()
        summary = pipeline.run_training(
            model_name="xgboost",
            model_params={"n_estimators": 20, "early_stopping_rounds": 5},
            apply_smote=False,
            n_synthetic=500,
            save_parquet=False,
        )
        assert summary["model"] == "xgboost"
        assert summary["test_metrics"]["auc_pr"] > 0.0
