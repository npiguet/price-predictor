"""Tests for the CLI eval subcommand (FR-007 through FR-010)."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest


FIXTURES_DIR = Path(__file__).parents[2] / "fixtures" / "forge_cards"


@pytest.fixture
def bolt_file(tmp_path: Path) -> Path:
    """Create a temporary Forge card script file."""
    f = tmp_path / "lightning_bolt.txt"
    f.write_text("Name:Lightning Bolt\nManaCost:R\nTypes:Instant\nOracle:Deals 3 damage.")
    return f


@pytest.fixture
def success_response() -> bytes:
    """JSON response bytes for a successful evaluation."""
    return json.dumps({
        "predicted_price_eur": 2.35,
        "model_version": "20260301-143000",
    }).encode("utf-8")


def _make_args(file: str, endpoint: str = "http://localhost:8000/api/v1/evaluate") -> argparse.Namespace:
    return argparse.Namespace(file=file, endpoint=endpoint)


class TestRunEvalSuccess:
    def test_displays_price_and_model_version(
        self, bolt_file: Path, success_response: bytes, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        mock_resp = MagicMock()
        mock_resp.read.return_value = success_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("price_predictor.infrastructure.cli.urlopen", return_value=mock_resp):
            exit_code = run_eval(_make_args(str(bolt_file)))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "2.35" in out
        assert "20260301-143000" in out

    def test_stdout_format_matches_spec(
        self, bolt_file: Path, success_response: bytes, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        mock_resp = MagicMock()
        mock_resp.read.return_value = success_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("price_predictor.infrastructure.cli.urlopen", return_value=mock_resp):
            run_eval(_make_args(str(bolt_file)))

        out = capsys.readouterr().out
        assert "Predicted price:" in out
        assert "Model version:" in out


class TestRunEvalFileErrors:
    def test_file_not_found_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        missing = tmp_path / "nonexistent.txt"
        exit_code = run_eval(_make_args(str(missing)))

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "File not found" in err

    def test_directory_path_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        exit_code = run_eval(_make_args(str(tmp_path)))

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "not a file" in err.lower() or "file" in err.lower()


class TestRunEvalEndpointErrors:
    def test_unreachable_endpoint_returns_2(
        self, bolt_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        with patch(
            "price_predictor.infrastructure.cli.urlopen",
            side_effect=URLError("Connection refused"),
        ):
            exit_code = run_eval(_make_args(str(bolt_file)))

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "could not connect" in err.lower() or "connect" in err.lower()

    def test_endpoint_400_error_returns_2(
        self, bolt_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        error_body = json.dumps({"error": "Failed to parse card script: No Types field"}).encode()
        http_error = HTTPError(
            url="http://localhost:8000/api/v1/evaluate",
            code=400,
            msg="Bad Request",
            hdrs=MagicMock(),
            fp=io.BytesIO(error_body),
        )

        with patch("price_predictor.infrastructure.cli.urlopen", side_effect=http_error):
            exit_code = run_eval(_make_args(str(bolt_file)))

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "400" in err
        assert "No Types field" in err

    def test_endpoint_500_error_returns_2(
        self, bolt_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        error_body = json.dumps({"error": "Prediction failed: model crashed"}).encode()
        http_error = HTTPError(
            url="http://localhost:8000/api/v1/evaluate",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),
            fp=io.BytesIO(error_body),
        )

        with patch("price_predictor.infrastructure.cli.urlopen", side_effect=http_error):
            exit_code = run_eval(_make_args(str(bolt_file)))

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "500" in err


class TestRunEvalEndpointFlag:
    def test_custom_endpoint_is_used(
        self, bolt_file: Path, success_response: bytes
    ) -> None:
        from price_predictor.infrastructure.cli import run_eval

        mock_resp = MagicMock()
        mock_resp.read.return_value = success_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("price_predictor.infrastructure.cli.urlopen", return_value=mock_resp) as mock_urlopen:
            run_eval(_make_args(str(bolt_file), endpoint="http://myhost:9000/api/v1/evaluate"))

        # Verify the request was sent to the custom endpoint
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert request_obj.full_url == "http://myhost:9000/api/v1/evaluate"


class TestEvalSubcommandParser:
    def test_eval_subcommand_registered(self) -> None:
        from price_predictor.infrastructure.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval", "card.txt"])
        assert args.command == "eval"
        assert args.file == "card.txt"

    def test_eval_default_endpoint(self) -> None:
        from price_predictor.infrastructure.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval", "card.txt"])
        assert args.endpoint == "http://localhost:8000/api/v1/evaluate"

    def test_eval_custom_endpoint(self) -> None:
        from price_predictor.infrastructure.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval", "card.txt", "--endpoint", "http://other:9000/api/v1/evaluate"])
        assert args.endpoint == "http://other:9000/api/v1/evaluate"
