"""Tests for the FastAPI prediction server endpoint."""

from __future__ import annotations

import json
import logging
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
        assert "sklearn" in data
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert isinstance(data["sklearn"]["model_version"], str)

    def test_partial_script_only_types_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Test Card\nTypes:Instant\n",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sklearn" in data
        assert "predicted_price_eur" in data["sklearn"]
        assert "model_version" in data["sklearn"]

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
        assert data["sklearn"]["predicted_price_eur"] > 0

    def test_response_content_type_is_json(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/evaluate",
            content="Name:Test\nTypes:Instant\n",
            headers={"Content-Type": "text/plain"},
        )
        assert "application/json" in response.headers["content-type"]


class TestStructuredRequestLogging:
    """Tests for FR-012: structured JSON request logging."""

    def test_success_log_contains_required_fields(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.infrastructure.server"):
            client.post(
                "/api/v1/evaluate",
                content="Name:Lightning Bolt\nManaCost:R\nTypes:Instant\nOracle:Deals 3 damage.",
                headers={"Content-Type": "text/plain"},
            )

        log_entries = [r for r in caplog.records if "evaluate_request" in r.getMessage()]
        assert len(log_entries) == 1
        entry = json.loads(log_entries[0].getMessage())
        assert entry["event"] == "evaluate_request"
        assert "timestamp" in entry
        assert entry["status_code"] == 200
        assert isinstance(entry["latency_ms"], (int, float))
        assert entry["latency_ms"] > 0
        assert entry["card_name"] == "Lightning Bolt"
        assert entry["card_types"] == ["Instant"]
        assert entry["card_mana_cost"] == "R"
        assert isinstance(entry["sklearn_predicted_price_eur"], float)
        assert entry["sklearn_model_version"] == "20260301-143000"

    def test_parse_error_log_contains_required_fields(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.infrastructure.server"):
            client.post(
                "/api/v1/evaluate",
                content="",
                headers={"Content-Type": "text/plain"},
            )

        log_entries = [r for r in caplog.records if "evaluate_request" in r.getMessage()]
        assert len(log_entries) == 1
        entry = json.loads(log_entries[0].getMessage())
        assert entry["event"] == "evaluate_request"
        assert "timestamp" in entry
        assert entry["status_code"] == 400
        assert isinstance(entry["latency_ms"], (int, float))
        assert entry["latency_ms"] >= 0
        assert "error" in entry

    def test_prediction_error_log_contains_required_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model crashed")

        mock_fe = MagicMock()
        mock_fe.transform.return_value = [[1, 2, 3]]

        artifact = {
            "model": mock_model,
            "feature_engineering": mock_fe,
            "model_version": "broken-model",
        }

        from price_predictor.infrastructure.server import create_app

        app = create_app(artifact)
        error_client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="price_predictor.infrastructure.server"):
            error_client.post(
                "/api/v1/evaluate",
                content="Name:Test Card\nTypes:Instant\n",
                headers={"Content-Type": "text/plain"},
            )

        log_entries = [r for r in caplog.records if "evaluate_request" in r.getMessage()]
        assert len(log_entries) == 1
        entry = json.loads(log_entries[0].getMessage())
        assert entry["event"] == "evaluate_request"
        assert entry["status_code"] == 500
        assert isinstance(entry["latency_ms"], (int, float))
        assert "error" in entry

    def test_log_timestamp_is_iso_format(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        from datetime import datetime

        with caplog.at_level(logging.INFO, logger="price_predictor.infrastructure.server"):
            client.post(
                "/api/v1/evaluate",
                content="Name:Test\nTypes:Instant\n",
                headers={"Content-Type": "text/plain"},
            )

        log_entries = [r for r in caplog.records if "evaluate_request" in r.getMessage()]
        entry = json.loads(log_entries[0].getMessage())
        # Should parse as ISO 8601 without error
        datetime.fromisoformat(entry["timestamp"])

    def test_partial_card_log_has_null_mana_cost(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.infrastructure.server"):
            client.post(
                "/api/v1/evaluate",
                content="Name:Island\nTypes:Basic Land Island\n",
                headers={"Content-Type": "text/plain"},
            )

        log_entries = [r for r in caplog.records if "evaluate_request" in r.getMessage()]
        entry = json.loads(log_entries[0].getMessage())
        assert entry["card_mana_cost"] is None
        assert entry["status_code"] == 200


class TestDualModelResponse:
    """Tests for US2: dual-model response format."""

    def test_both_models_present(self):
        """When both models loaded, response has sklearn and transformer objects."""
        import numpy as np
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient
        from price_predictor.infrastructure.server import create_app

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([np.log(2.35)])
        mock_fe = MagicMock()
        mock_fe.transform.return_value = [[1, 2, 3]]
        sklearn_artifact = {"model": mock_model, "feature_engineering": mock_fe, "model_version": "20260301-143000"}

        # Mock transformer that returns a tensor
        import torch
        mock_transformer = MagicMock()
        mock_transformer.return_value = torch.tensor([0.85])  # shifted-log value
        mock_transformer.eval = MagicMock()
        mock_transformer.to = MagicMock(return_value=mock_transformer)
        # Mock parameters() to return a tensor so next(...).device works
        param = torch.zeros(1)
        mock_transformer.parameters = MagicMock(return_value=iter([param]))
        mock_config = MagicMock()
        mock_config.max_seq_len = 64

        transformer_artifact = {"model": mock_transformer, "config": mock_config, "model_version": "transformer-v1"}

        app = create_app(sklearn_artifact, transformer_artifact=transformer_artifact)
        client = TestClient(app)

        response = client.post(
            "/api/v1/evaluate",
            content="Name:Lightning Bolt\nManaCost:R\nTypes:Instant\nOracle:Deals 3 damage.",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sklearn" in data
        assert "transformer" in data
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)
        assert isinstance(data["transformer"]["predicted_price_eur"], float)
        assert data["sklearn"]["model_version"] == "20260301-143000"
        assert data["transformer"]["model_version"] == "transformer-v1"

    def test_transformer_null_when_unavailable(self):
        """When no transformer loaded, transformer is null."""
        import numpy as np
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient
        from price_predictor.infrastructure.server import create_app

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([np.log(2.35)])
        mock_fe = MagicMock()
        mock_fe.transform.return_value = [[1, 2, 3]]
        sklearn_artifact = {"model": mock_model, "feature_engineering": mock_fe, "model_version": "20260301-143000"}

        app = create_app(sklearn_artifact, transformer_artifact=None)
        client = TestClient(app)

        response = client.post(
            "/api/v1/evaluate",
            content="Name:Lightning Bolt\nManaCost:R\nTypes:Instant\nOracle:Deals 3 damage.",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sklearn" in data
        assert data["transformer"] is None
        assert isinstance(data["sklearn"]["predicted_price_eur"], float)

    def test_response_schema_structure(self):
        """Response must have exactly sklearn and transformer keys at top level."""
        import numpy as np
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient
        from price_predictor.infrastructure.server import create_app

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([np.log(1.0)])
        mock_fe = MagicMock()
        mock_fe.transform.return_value = [[1]]
        artifact = {"model": mock_model, "feature_engineering": mock_fe, "model_version": "test"}

        app = create_app(artifact)
        client = TestClient(app)

        response = client.post(
            "/api/v1/evaluate",
            content="Name:Test\nTypes:Instant\n",
            headers={"Content-Type": "text/plain"},
        )
        data = response.json()
        assert set(data.keys()) == {"sklearn", "transformer"}
