import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor, TARGET


@pytest.fixture
def raw_df():
    ingestion = DataIngestion(raw_data_dir="data/raw")
    return ingestion.generate_synthetic(n_samples=500, fraud_rate=0.05, save=False)


class TestDataIngestion:
    def test_generate_synthetic_shape(self, raw_df):
        assert len(raw_df) == 500
        assert TARGET in raw_df.columns

    def test_generate_synthetic_fraud_rate(self, raw_df):
        fraud_count = raw_df[TARGET].sum()
        # 5% of 500 = 25, allow ±15
        assert 10 <= fraud_count <= 40, f"Unexpected fraud count: {fraud_count}"

    def test_required_columns_present(self, raw_df):
        for col in ("TransactionID", "TransactionAmt", "TransactionDT", "ProductCD", "isFraud"):
            assert col in raw_df.columns

    def test_validate_schema_passes(self, raw_df):
        ingestion = DataIngestion()
        assert ingestion.validate_schema(raw_df) is True

    def test_validate_schema_fails_on_missing_col(self, raw_df):
        ingestion = DataIngestion()
        df_bad = raw_df.drop(columns=["TransactionAmt"])
        with pytest.raises(ValueError, match="Missing required columns"):
            ingestion.validate_schema(df_bad)


class TestDataPreprocessor:
    def test_clean_clips_negative_amount(self, raw_df):
        preprocessor = DataPreprocessor()
        df = raw_df.copy()
        df.loc[0, "TransactionAmt"] = -100
        cleaned = preprocessor._clean(df)
        assert cleaned.loc[0, "TransactionAmt"] == 0

    def test_missing_flag_created(self):
        """Columns with > 10% missing should get a _was_missing binary flag."""
        df = pd.DataFrame({
            "TransactionID": range(100),
            "TransactionAmt": np.random.rand(100) * 100 + 1,
            "TransactionDT": np.random.randint(0, 86400, 100),
            "ProductCD": ["W"] * 100,
            "DeviceType": [None] * 30 + ["desktop"] * 70,  # 30% missing
            "isFraud": [0] * 95 + [1] * 5,
        })
        preprocessor = DataPreprocessor()
        preprocessor._handle_missing(df, fit=True)
        assert "DeviceType" in preprocessor.cols_to_flag

    def test_fit_transform_returns_correct_shapes(self, raw_df):
        preprocessor = DataPreprocessor()
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocessor.fit_transform(raw_df)
        total = len(X_tr) + len(X_val) + len(X_te)
        assert total == len(raw_df)
        assert X_tr.shape[1] == X_val.shape[1] == X_te.shape[1]

    def test_target_not_in_features(self, raw_df):
        preprocessor = DataPreprocessor()
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocessor.fit_transform(raw_df)
        assert TARGET not in X_tr.columns

    def test_transaction_id_not_in_features(self, raw_df):
        preprocessor = DataPreprocessor()
        X_tr, *_ = preprocessor.fit_transform(raw_df)
        assert "TransactionID" not in X_tr.columns

    def test_scaled_amount_near_zero_mean(self, raw_df):
        preprocessor = DataPreprocessor()
        from src.features.build_features import FeatureEngineer
        df = FeatureEngineer().transform(raw_df)
        X_tr, *_ = preprocessor.fit_transform(df)
        if "TransactionAmt" in X_tr.columns:
            assert abs(X_tr["TransactionAmt"].mean()) < 0.5
