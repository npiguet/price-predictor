"""Unit tests for PoolConnector subprocess error propagation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.pool_connector import PoolConnector


class TestPoolConnectorErrorPropagation:
    def test_non_zero_exit_code_raises_runtime_error(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch.object(connector, "_resolve_jar_path", return_value=Path("fake.jar")):
                with pytest.raises(RuntimeError):
                    connector.generate("RVR", 10, tmp_path / "pools")

    def test_exit_code_zero_does_not_raise(self, tmp_path):
        connector = PoolConnector()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            with patch.object(connector, "_resolve_jar_path", return_value=Path("fake.jar")):
                connector.generate("RVR", 10, tmp_path / "pools")

    def test_java_not_found_raises_file_not_found(self, tmp_path):
        connector = PoolConnector()

        with patch.object(connector, "_resolve_jar_path", return_value=Path("fake.jar")):
            with patch("subprocess.run", side_effect=FileNotFoundError("java not found")):
                with pytest.raises(FileNotFoundError):
                    connector.generate("RVR", 10, tmp_path / "pools")

    def test_generate_raises_if_jar_not_found(self, tmp_path):
        connector = PoolConnector()
        # Do not mock _resolve_jar_path — let it raise if the JAR doesn't exist
        with patch.object(connector, "_resolve_jar_path",
                          side_effect=FileNotFoundError("JAR not found")):
            with pytest.raises(FileNotFoundError):
                connector.generate("RVR", 10, tmp_path / "pools")
