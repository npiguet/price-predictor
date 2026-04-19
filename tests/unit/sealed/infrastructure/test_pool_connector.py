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
