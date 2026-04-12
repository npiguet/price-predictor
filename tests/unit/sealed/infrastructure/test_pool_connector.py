"""Unit tests for PoolConnector subprocess error propagation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.pool_connector import PoolConnector


@pytest.fixture
def stub_classpath():
    """Stub build_forge_classpath at the consumer site to avoid real JAR resolution."""
    with patch(
        "sealed.infrastructure.pool_connector.build_forge_classpath",
        return_value="cp",
    ):
        yield


class TestPoolConnectorErrorPropagation:
    def test_non_zero_exit_code_raises_runtime_error(self, tmp_path, stub_classpath):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_exit_code_zero_does_not_raise(self, tmp_path, stub_classpath):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            connector.generate("RVR", 10, tmp_path / "pools")

    def test_java_not_found_raises_file_not_found(self, tmp_path, stub_classpath):
        connector = PoolConnector()

        with patch("subprocess.run", side_effect=FileNotFoundError("java not found")):
            with pytest.raises(FileNotFoundError):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_generate_raises_if_jar_not_found(self, tmp_path):
        connector = PoolConnector()
        with patch(
            "sealed.infrastructure.pool_connector.build_forge_classpath",
            side_effect=FileNotFoundError("JAR not found"),
        ):
            with pytest.raises(FileNotFoundError):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_command_includes_pool_main_and_args(self, tmp_path, stub_classpath):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            connector.generate("RVR", 10, tmp_path / "pools")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "java"
        assert "com.pricepredictor.connector.PoolMain" in cmd
        assert "--set" in cmd and "RVR" in cmd
        assert "--size" in cmd and "10" in cmd
