"""Unit tests for PoolConnector subprocess error propagation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.pool_connector import PoolConnector

_RFW = "sealed.infrastructure.pool_connector.run_forge_worker"


class TestPoolConnectorErrorPropagation:
    def test_non_zero_exit_code_raises_runtime_error(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch(_RFW, return_value=mock_result):
            with pytest.raises(RuntimeError):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_exit_code_zero_does_not_raise(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch(_RFW, return_value=mock_result):
            connector.generate("RVR", 10, tmp_path / "pools")

    def test_java_not_found_raises_file_not_found(self, tmp_path):
        connector = PoolConnector()

        with patch(_RFW, side_effect=FileNotFoundError("java not found")):
            with pytest.raises(FileNotFoundError):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_command_includes_pool_main_and_args(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch(_RFW, return_value=mock_result) as mock_run:
            connector.generate("RVR", 10, tmp_path / "pools")

        assert mock_run.call_args[0][0] == (
            "com.pricepredictor.connector.PoolMain"
        )
        args = mock_run.call_args.kwargs["main_args"]
        assert "--set" in args and "RVR" in args
        assert "--size" in args and "10" in args

    def test_set_code_none_omits_set_argument(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch(_RFW, return_value=mock_result) as mock_run:
            connector.generate(None, 10, tmp_path / "pools")

        args = mock_run.call_args.kwargs["main_args"]
        assert "--set" not in args
        assert "--size" in args and "10" in args
        assert "--pools-path" in args


class TestDepletedPools:
    """`--exclude-cards` reaches the JVM, and stays absent without it (T165).

    A depleted pool is how the effect model's training corpus avoids holding
    held-out cards. The flag going missing is silent from this side: pools
    generate fine, and the cost only shows up hours later as a training corpus
    whose games are nearly all tainted.
    """

    def test_the_exclusion_file_reaches_the_worker(self, tmp_path):
        connector = PoolConnector()
        result = MagicMock()
        result.returncode = 0
        listed = tmp_path / "holdout-cards.txt"

        with patch(_RFW, return_value=result) as run:
            connector.generate("RVR", 10, tmp_path / "pools", listed)

        args = run.call_args.kwargs["main_args"]
        assert "--exclude-cards" in args
        assert args[args.index("--exclude-cards") + 1] == str(listed)

    def test_an_ordinary_run_passes_no_exclusion(self, tmp_path):
        connector = PoolConnector()
        result = MagicMock()
        result.returncode = 0

        with patch(_RFW, return_value=result) as run:
            connector.generate("RVR", 10, tmp_path / "pools")

        assert "--exclude-cards" not in run.call_args.kwargs["main_args"]
