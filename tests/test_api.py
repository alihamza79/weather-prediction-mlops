"""Tests for FastAPI prediction service."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Weather Prediction API"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "weather_prediction" in response.text or "python" in response.text

    def test_predict_without_model(self, client):
        """Test prediction fails gracefully without model."""
        response = client.post("/predict", json={
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "temperature": 20.0,
            "humidity": 65.0,
            "pressure": 1013.0,
            "wind_speed": 5.0,
        })
        # Should return 503 if model not loaded
        assert response.status_code in [200, 503]

    def test_predict_validation(self, client):
        """Test input validation."""
        # Invalid temperature (out of range)
        response = client.post("/predict", json={
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "temperature": 100.0,  # Invalid: > 60
            "humidity": 65.0,
            "pressure": 1013.0,
            "wind_speed": 5.0,
        })
        assert response.status_code == 422  # Validation error

    def test_predict_missing_required_fields(self, client):
        """Test that missing required fields return error."""
        response = client.post("/predict", json={
            "hour_sin": 0.5,
            # Missing other required fields
        })
        assert response.status_code == 422

    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint."""
        response = client.post("/predict/batch", json={
            "features": [
                {
                    "hour_sin": 0.5,
                    "hour_cos": 0.866,
                    "temperature": 20.0,
                    "humidity": 65.0,
                    "pressure": 1013.0,
                    "wind_speed": 5.0,
                },
                {
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "temperature": 22.0,
                    "humidity": 60.0,
                    "pressure": 1015.0,
                    "wind_speed": 3.0,
                },
            ]
        })
        # Should return 503 if model not loaded, 200 if loaded
        assert response.status_code in [200, 503]

