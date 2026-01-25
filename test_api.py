# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
