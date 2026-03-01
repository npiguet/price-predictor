"""Integration tests for the eval CLI subcommand with a real trained model."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from price_predictor.application.train import TrainModelUseCase
from price_predictor.infrastructure.model_store import load_model
from price_predictor.infrastructure.server import create_app

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a real model for integration tests."""
    tmp = tmp_path_factory.mktemp("models")
    use_case = TrainModelUseCase()
    use_case.execute(
        forge_cards_path=FIXTURES_DIR / "forge_cards",
        prices_path=FIXTURES_DIR / "allprices_sample.json",
        printings_path=FIXTURES_DIR / "allprintings_sample.json",
        output_path=tmp,
        test_split=0.2,
        random_seed=42,
    )
    return tmp


@pytest.fixture(scope="module")
def model_path(trained_model_dir: Path) -> Path:
    return trained_model_dir / "latest.joblib"


@pytest.fixture(scope="module")
def integration_client(model_path: Path) -> TestClient:
    """Create a TestClient with a real trained model."""
    artifact = load_model(model_path)
    artifact["model_version"] = model_path.stem
    app = create_app(artifact)
    return TestClient(app)


def _fake_urlopen(test_client: TestClient):
    """Create a urlopen replacement that routes to the TestClient."""

    def _urlopen(request, **kwargs):
        body = request.data.decode("utf-8") if isinstance(request.data, bytes) else request.data
        response = test_client.post(
            "/api/v1/evaluate",
            content=body,
            headers={"Content-Type": "text/plain"},
        )

        result = io.BytesIO(response.content)
        result.status = response.status_code
        result.read = lambda: response.content

        if response.status_code >= 400:
            from urllib.error import HTTPError
            raise HTTPError(
                url=request.full_url,
                code=response.status_code,
                msg="Error",
                hdrs=None,
                fp=io.BytesIO(response.content),
            )

        # Make it work as context manager
        result.__enter__ = lambda s: s
        result.__exit__ = lambda s, *a: None
        return result

    return _urlopen


@pytest.mark.integration
class TestEvalIntegration:
    def test_valid_card_file_displays_price(
        self,
        integration_client: TestClient,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        card_file = FIXTURES_DIR / "forge_cards" / "lightning_bolt.txt"
        args = argparse.Namespace(
            file=str(card_file),
            endpoint="http://localhost:8000/api/v1/evaluate",
        )

        with patch(
            "price_predictor.infrastructure.cli.urlopen",
            side_effect=_fake_urlopen(integration_client),
        ):
            exit_code = run_eval(args)

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Predicted price:" in out
        assert "Model version:" in out
        # Price should be a positive number
        price_line = [l for l in out.splitlines() if "Predicted price:" in l][0]
        price_str = price_line.split("€")[1].strip()
        assert float(price_str) > 0

    def test_malformed_file_relays_endpoint_error(
        self,
        integration_client: TestClient,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        bad_file = tmp_path / "bad_card.txt"
        bad_file.write_text("This is not a valid Forge card script")
        args = argparse.Namespace(
            file=str(bad_file),
            endpoint="http://localhost:8000/api/v1/evaluate",
        )

        with patch(
            "price_predictor.infrastructure.cli.urlopen",
            side_effect=_fake_urlopen(integration_client),
        ):
            exit_code = run_eval(args)

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "400" in err

    def test_cli_price_matches_direct_endpoint_price(
        self,
        integration_client: TestClient,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SC-006: CLI price must be identical to direct endpoint price."""
        from price_predictor.infrastructure.cli import run_eval

        card_file = FIXTURES_DIR / "forge_cards" / "lightning_bolt.txt"
        card_text = card_file.read_text(encoding="utf-8")

        # Get price directly from endpoint
        direct_response = integration_client.post(
            "/api/v1/evaluate",
            content=card_text,
            headers={"Content-Type": "text/plain"},
        )
        direct_price = direct_response.json()["predicted_price_eur"]

        # Get price via CLI
        args = argparse.Namespace(
            file=str(card_file),
            endpoint="http://localhost:8000/api/v1/evaluate",
        )

        with patch(
            "price_predictor.infrastructure.cli.urlopen",
            side_effect=_fake_urlopen(integration_client),
        ):
            run_eval(args)

        out = capsys.readouterr().out
        price_line = [l for l in out.splitlines() if "Predicted price:" in l][0]
        cli_price = float(price_line.split("€")[1].strip())

        assert cli_price == direct_price
