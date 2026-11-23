
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from src.api import app


EXPECTED_FEATURES = [
    "mean radius",
    "mean texture",
]


class FakeModel:
    """A lightweight model for API tests."""

    def predict(self, dataframe):
        assert list(dataframe.columns) == EXPECTED_FEATURES
        return [0]


@asynccontextmanager
async def mock_lifespan(app):
    """Provide mocked model resources without connecting to MLflow."""

    app.state.model = FakeModel()
    app.state.model_name = "breast-cancer-classifier"
    app.state.model_version = "2"
    app.state.model_alias = "champion"
    app.state.feature_names = EXPECTED_FEATURES

    yield


# Replace production lifespan for isolated API tests.
app.router.lifespan_context = mock_lifespan


@pytest.fixture
def client():
    """Create a test client and manage the application lifespan."""

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_name": "breast-cancer-classifier",
        "model_version": "2",
        "model_alias": "champion",
    }


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": 14.2,
                "mean texture": 20.1,
            }
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["class_label"] == "benign"
    assert result["model_version"] == "2"


def test_missing_features(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": 14.2,
            }
        },
    )

    assert response.status_code == 422
    assert (
        "mean texture"
        in response.json()["detail"]["missing_features"]
    )


def test_extra_features(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": 14.2,
                "mean texture": 20.1,
                "unknown feature": 10.0,
            }
        },
    )

    assert response.status_code == 422
    assert (
        "unknown feature"
        in response.json()["detail"]["extra_features"]
    )


def test_nan_values(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": "NaN",
                "mean texture": 20.1,
            }
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Feature values must all be finite."
    )


def test_infinite_values(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": "Infinity",
                "mean texture": 20.1,
            }
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Feature values must all be finite."
    )
