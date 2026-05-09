"""
Tests for src/api/main.py FastAPI endpoints.

Uses httpx.AsyncClient via TestClient — no running server needed.
The champion model (models/lgbm_champion.pkl) is required for /predict tests;
health/model-info tests work even without a model loaded.
"""
import json
import os
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Import the app — startup event fires on TestClient context
# ---------------------------------------------------------------------------
from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_keys(self):
        r = client.get("/health")
        body = r.json()
        for key in ["status", "model_loaded", "threshold", "n_features", "model_version"]:
            assert key in body, f"Missing key: {key}"

    def test_health_status_is_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /model-info (only meaningful if champion model exists)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists("models/lgbm_champion.pkl"),
    reason="Champion model not found — run training pipeline first",
)
class TestModelInfo:
    def test_model_info_returns_200(self):
        r = client.get("/model-info")
        assert r.status_code == 200

    def test_model_info_keys(self):
        r = client.get("/model-info")
        body = r.json()
        for key in ["model_type", "threshold", "n_features", "dataset", "version"]:
            assert key in body


# ---------------------------------------------------------------------------
# /predict (requires champion model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists("models/lgbm_champion.pkl"),
    reason="Champion model not found — run training pipeline first",
)
class TestPredict:
    LEGIT_TXN = {
        "TransactionAmt": 25.0,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "TransactionID": 1001,
    }

    SUSPICIOUS_TXN = {
        "TransactionAmt": 9999.0,
        "card4": "discover",
        "DeviceType": None,          # missing DeviceType is a fraud signal
        "P_emaildomain": "anonymous.com",
        "TransactionID": 1002,
    }

    def test_predict_returns_200(self):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.status_code == 200

    def test_predict_response_keys(self):
        r = client.post("/predict", json=self.LEGIT_TXN)
        body = r.json()
        for key in ["fraud_probability", "is_fraud", "threshold_used", "risk_level",
                    "top_shap_features", "model_version"]:
            assert key in body, f"Missing key: {key}"

    def test_predict_probability_in_range(self):
        r = client.post("/predict", json=self.LEGIT_TXN)
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_level_valid(self):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.json()["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "VERY_LOW"}

    def test_predict_transaction_id_echoed(self):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.json()["TransactionID"] == 1001

    def test_predict_invalid_amount_returns_422(self):
        r = client.post("/predict", json={"TransactionAmt": -50.0})
        assert r.status_code == 422

    def test_predict_missing_required_field_returns_422(self):
        r = client.post("/predict", json={"card4": "visa"})
        assert r.status_code == 422

    def test_predict_batch_returns_correct_count(self):
        payload = {"transactions": [self.LEGIT_TXN, self.SUSPICIOUS_TXN]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_predict_batch_all_probabilities_valid(self):
        payload = {"transactions": [self.LEGIT_TXN, self.SUSPICIOUS_TXN]}
        r = client.post("/predict/batch", json=payload)
        for pred in r.json()["predictions"]:
            assert 0.0 <= pred["fraud_probability"] <= 1.0
