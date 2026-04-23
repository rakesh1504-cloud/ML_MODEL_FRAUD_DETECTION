import pytest
import pandas as pd
import numpy as np

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
        assert 10 <= fraud_count <= 40, f"Unexpected fraud count: {fraud_count}"

    def test_validate_schema_passes(self, raw_df):
        ingestion = DataIngestion()
        assert ingestion.validate_schema(raw_df) is True

    def test_validate_schema_fails_on_missing_col(self, raw_df):
        ingestion = DataIngestion()
        df_bad = raw_df.drop(columns=["amount"])
        with pytest.raises(ValueError, match="Missing required columns"):
            ingestion.validate_schema(df_bad)


class TestDataPreprocessor:
    def test_clean_removes_negatives(self, raw_df):
        preprocessor = DataPreprocessor()
        df = raw_df.copy()
        df.loc[0, "amount"] = -100
        cleaned = preprocessor.clean(df)
        assert cleaned.loc[0, "amount"] == 0

    def test_clean_clips_hour(self, raw_df):
        preprocessor = DataPreprocessor()
        df = raw_df.copy()
        df.loc[0, "hour"] = 99
        cleaned = preprocessor.clean(df)
        assert cleaned.loc[0, "hour"] == 23

    def test_encode_categoricals(self, raw_df):
        preprocessor = DataPreprocessor()
        encoded = preprocessor.encode_categoricals(raw_df.copy(), fit=True)
        assert encoded["merchant_category"].dtype in [np.int32, np.int64, int]

    def test_fit_transform_returns_correct_shapes(self, raw_df):
        preprocessor = DataPreprocessor()
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocessor.fit_transform(raw_df)
        total = len(X_tr) + len(X_val) + len(X_te)
        assert total == len(raw_df)
        assert X_tr.shape[1] == X_val.shape[1] == X_te.shape[1]

    def test_scale_features_zero_mean(self, raw_df):
        preprocessor = DataPreprocessor()
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocessor.fit_transform(raw_df)
        means = X_tr["amount"].mean()
        assert abs(means) < 0.1, f"Expected near-zero mean after scaling, got {means:.4f}"
