"""
Tests for src/api/main.py FastAPI endpoints.

Uses TestClient as a context manager to trigger the lifespan startup event,
which loads the champion model. Tests that require the model are skipped if
models/lgbm_champion.pkl does not exist.
"""
import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_EXISTS = os.path.exists("models/lgbm_champion.pkl")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health  (works with or without a model loaded)
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_keys(self, client):
        r = client.get("/health")
        body = r.json()
        for key in ["status", "model_loaded", "threshold", "n_features", "model_version"]:
            assert key in body, f"Missing key: {key}"

    def test_health_status_is_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /model-info  (requires champion model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="Champion model not found — run training pipeline first")
class TestModelInfo:
    def test_model_info_returns_200(self, client):
        r = client.get("/model-info")
        assert r.status_code == 200

    def test_model_info_keys(self, client):
        r = client.get("/model-info")
        body = r.json()
        for key in ["model_type", "threshold", "n_features", "dataset", "version"]:
            assert key in body


# ---------------------------------------------------------------------------
# /predict  (requires champion model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="Champion model not found — run training pipeline first")
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
        "DeviceType": None,
        "P_emaildomain": "anonymous.com",
        "TransactionID": 1002,
    }

    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.status_code == 200

    def test_predict_response_keys(self, client):
        r = client.post("/predict", json=self.LEGIT_TXN)
        body = r.json()
        for key in ["fraud_probability", "is_fraud", "threshold_used", "risk_level",
                    "top_shap_features", "model_version"]:
            assert key in body, f"Missing key: {key}"

    def test_predict_probability_in_range(self, client):
        r = client.post("/predict", json=self.LEGIT_TXN)
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_level_valid(self, client):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.json()["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "VERY_LOW"}

    def test_predict_transaction_id_echoed(self, client):
        r = client.post("/predict", json=self.LEGIT_TXN)
        assert r.json()["TransactionID"] == 1001

    def test_predict_invalid_amount_returns_422(self, client):
        r = client.post("/predict", json={"TransactionAmt": -50.0})
        assert r.status_code == 422

    def test_predict_missing_required_field_returns_422(self, client):
        r = client.post("/predict", json={"card4": "visa"})
        assert r.status_code == 422

    def test_predict_batch_returns_correct_count(self, client):
        payload = {"transactions": [self.LEGIT_TXN, self.SUSPICIOUS_TXN]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_predict_batch_all_probabilities_valid(self, client):
        payload = {"transactions": [self.LEGIT_TXN, self.SUSPICIOUS_TXN]}
        r = client.post("/predict/batch", json=payload)
        for pred in r.json()["predictions"]:
            assert 0.0 <= pred["fraud_probability"] <= 1.0
