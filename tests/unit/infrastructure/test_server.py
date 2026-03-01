"""Tests for the FastAPI prediction server endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model_artifact() -> dict:
    """Create a mock model artifact that returns a fixed prediction."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([np.log(2.35)])

    mock_fe = MagicMock()
    mock_fe.transform.return_value = [[1, 2, 3]]

    return {
        "model": mock_model,
        "feature_engineering": mock_fe,
        "model_version": "20260301-143000",
    }


@pytest.fixture
def client(mock_model_artifact: dict) -> TestClient:
    from price_predictor.infrastructure.server import create_app

    app = create_app(mock_model_artifact)
    return TestClient(app)


class TestEvaluateEndpoint:
    def test_valid_complete_forge_script_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Lightning Bolt\nManaCost:R\nTypes:Instant\nOracle:Deals 3 damage.",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price_eur" in data
        assert isinstance(data["predicted_price_eur"], float)
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_partial_script_only_types_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Test Card\nTypes:Instant\n",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price_eur" in data
        assert "model_version" in data

    def test_empty_body_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_unparseable_body_no_types_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Bad Card\nManaCost:R\n",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "types" in data["error"].lower()

    def test_made_up_card_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content=(
                "Name:Imaginary Dragon\nManaCost:4 R R\n"
                "Types:Creature Dragon\nPT:6/6\nK:Flying\nK:Haste"
            ),
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_price_eur"] > 0

    def test_response_content_type_is_json(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Test\nTypes:Instant\n",
            headers={"Content-Type": "text/plain"},
        )
        assert "application/json" in response.headers["content-type"]
