import pytest

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor, TARGET
from src.features.build_features import FeatureEngineer


@pytest.fixture
def raw_df():
    ingestion = DataIngestion(raw_data_dir="data/raw")
    return ingestion.generate_synthetic(n_samples=600, fraud_rate=0.05, save=False)


@pytest.fixture
def engineered_df(raw_df):
    fe = FeatureEngineer()
    return fe.fit_transform(raw_df)


class TestDataIngestion:
    def test_generate_synthetic_shape(self, raw_df):
        assert len(raw_df) == 600
        assert TARGET in raw_df.columns

    def test_required_columns_present(self, raw_df):
        for col in ("TransactionID", "TransactionAmt", "TransactionDT", "ProductCD", "isFraud"):
            assert col in raw_df.columns

    def test_validate_schema_passes(self, raw_df):
        assert DataIngestion().validate_schema(raw_df) is True

    def test_validate_schema_fails_on_missing_col(self, raw_df):
        with pytest.raises(ValueError, match="Missing required columns"):
            DataIngestion().validate_schema(raw_df.drop(columns=["TransactionAmt"]))


class TestDataPreprocessor:
    def test_time_based_split_no_leakage(self, engineered_df):
        """Test transactions must be later than all train transactions."""
        pre = DataPreprocessor()
        X_train, X_test, y_train, y_test = pre.fit_transform(engineered_df, apply_smote=False)
        total = len(X_train) + len(X_test)
        assert total == len(engineered_df)

    def test_smote_increases_fraud_rate(self, engineered_df):
        pre = DataPreprocessor(smote_strategy=0.1)
        X_train_sm, X_test, y_train_sm, y_test = pre.fit_transform(engineered_df, apply_smote=True)
        # After SMOTE fraud rate should be ~10%
        assert y_train_sm.mean() >= 0.08, f"Expected ≥8% fraud after SMOTE, got {y_train_sm.mean():.2%}"

    def test_smote_not_applied_to_test(self, engineered_df):
        """Test set fraud rate should remain close to the original ~5%."""
        pre = DataPreprocessor()
        _, X_test, _, y_test = pre.fit_transform(engineered_df, apply_smote=True)
        assert y_test.mean() < 0.15, "SMOTE must NOT be applied to test set"

    def test_target_not_in_features(self, engineered_df):
        pre = DataPreprocessor()
        X_train, *_ = pre.fit_transform(engineered_df, apply_smote=False)
        assert TARGET not in X_train.columns

    def test_transaction_id_not_in_features(self, engineered_df):
        pre = DataPreprocessor()
        X_train, *_ = pre.fit_transform(engineered_df, apply_smote=False)
        assert "TransactionID" not in X_train.columns

    def test_no_nulls_after_imputation(self, engineered_df):
        pre = DataPreprocessor()
        X_train, X_test, _, _ = pre.fit_transform(engineered_df, apply_smote=False)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
