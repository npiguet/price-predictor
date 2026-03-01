"""Tests for the serve CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from price_predictor.infrastructure.cli import build_parser, run_serve


class TestServeSubcommand:
    def test_serve_subcommand_is_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"

    def test_model_path_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.model_path == "models/latest.joblib"

    def test_host_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.host == "0.0.0.0"

    def test_port_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.port == 8000

    def test_custom_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "serve",
            "--model-path", "models/custom.joblib",
            "--host", "127.0.0.1",
            "--port", "9000",
        ])
        assert args.model_path == "models/custom.joblib"
        assert args.host == "127.0.0.1"
        assert args.port == 9000


class TestRunServe:
    def test_exits_2_when_model_not_found(self, capsys: pytest.CaptureFixture) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--model-path", "nonexistent/model.joblib"])
        exit_code = run_serve(args)
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    @patch("uvicorn.run")
    @patch("price_predictor.infrastructure.model_store.load_model")
    def test_calls_uvicorn_run_on_success(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_load.return_value = {
            "model": MagicMock(),
            "feature_engineering": MagicMock(),
        }

        parser = build_parser()
        args = parser.parse_args([
            "serve",
            "--model-path", str(tmp_path / "test_model.joblib"),
            "--host", "127.0.0.1",
            "--port", "9090",
        ])
        exit_code = run_serve(args)
        assert exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["host"] == "127.0.0.1"
        assert call_kwargs.kwargs["port"] == 9090
