import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api_service"))


@pytest.fixture
def client():
    mock_db_response = MagicMock()
    mock_db_response.status_code = 200
    mock_db_response.json.return_value = {
    "predictions": [
        {
        "hate speech": 0.1363648772239685,
        "offensive language": 0.2835945188999176,
        "neither": 0.5800406336784363
        }
    ]
    }

    mock_s3_client = MagicMock()
    mock_s3_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

    with patch("app.requests.post", return_value=mock_db_response), \
         patch("app.boto3.client", return_value=mock_s3_client):

        try:
            from app import app
        except ImportError:
            pytest.skip("Flask app module not found at api_service/app.py")

        app.config["TESTING"] = True

        with app.test_client() as client:
            yield client



class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/internal/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/internal/health")
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_health_contains_status(self, client):
        response = client.get("/internal/health")
        data = json.loads(response.data)
        assert "status" in data



class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/internal/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/internal/metrics")
        assert "text" in response.content_type


class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_input(self, client):
        payload = {"text": "This is a test sentence for classification."}
        response = client.post("/api", json=payload)
        assert response.status_code == 200

    def test_predict_response_contains_required_fields(self, client):
        payload = {"text": "Some input text."}
        response = client.post("/api", json=payload)
        data = json.loads(response.data)

        assert "text" in data
        assert "prediction" in data
        assert "confidence_score" in data

    def test_predict_echoes_input_text(self, client):
        input_text = "The quick brown fox jumps."
        payload = {"text": input_text}

        response = client.post("/api", json=payload)
        data = json.loads(response.data)

        assert data["text"] == input_text

    def test_predict_confidence_score_is_float_in_valid_range(self, client):
        payload = {"text": "Test text for confidence check."}
        response = client.post("/api", json=payload)
        data = json.loads(response.data)

        assert isinstance(data["confidence_score"], float)
        assert 0.0 <= data["confidence_score"] <= 1.0


    def test_predict_missing_text_field_returns_error(self, client):
        payload = {"wrong_field": "something"}
        response = client.post("/api", json=payload)

        assert 400 <= response.status_code < 500

    def test_predict_empty_body_returns_error(self, client):
        response = client.post("/api", data="", content_type="application/json")
        assert response.status_code in (400, 422, 500)
