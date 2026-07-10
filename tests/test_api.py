"""
Tests for the FastAPI serving layer. Uses FastAPI's TestClient (in-process,
no real socket needed) rather than spinning up a live uvicorn server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture
def client():
    """`with` form is required so FastAPI's lifespan (model loading) actually
    runs — a bare TestClient(app) does not trigger startup in this FastAPI
    version, which is an easy trap when converting curl smoke tests to
    pytest."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


def test_predict_valid_payload_returns_expected_shape(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Disease", "No disease")
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_used" in body


def test_predict_rejects_missing_fields(client):
    incomplete = {"age": 63, "sex": 1}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422  # FastAPI/pydantic validation error


def test_predict_rejects_wrong_type(client):
    bad = dict(VALID_PAYLOAD)
    bad["age"] = "not-a-number"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_sex(client):
    bad = dict(VALID_PAYLOAD)
    bad["sex"] = 7  # only 0/1 allowed
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint_exposed_for_prometheus(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests" in resp.content or b"# HELP" in resp.content
