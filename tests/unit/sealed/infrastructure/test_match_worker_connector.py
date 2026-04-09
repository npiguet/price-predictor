"""Unit tests for MatchWorkerConnector subprocess construction."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.match_worker_connector import MatchWorkerConnector


SEP = ";" if platform.system() == "Windows" else ":"


class FakeJar:
    """Returns predictable fake paths for JAR resolution."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def connector_jar(self) -> Path:
        jar = self.tmp_path / "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
        jar.touch()
        return jar

    def forge_dir(self) -> Path:
        return self.tmp_path / "forge"


class TestMatchWorkerConnectorCommandConstruction:
    def test_command_starts_with_java(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "java"

    def test_command_includes_xmx_flag(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        assert "-Xmx1200m" in cmd

    def test_command_includes_main_class(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        assert "com.pricepredictor.connector.MatchWorkerMain" in cmd

    def test_command_includes_classpath_flag(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        assert "-cp" in cmd

    def test_classpath_includes_connector_jar(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        jar = fake.connector_jar()
        with patch.object(connector, "_resolve_jar_path", return_value=jar):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        cp_index = cmd.index("-cp")
        classpath = cmd[cp_index + 1]
        assert str(jar) in classpath

    def test_classpath_includes_forge_gui_jar(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        cp_index = cmd.index("-cp")
        classpath = cmd[cp_index + 1]
        assert "forge-gui-2.0.10-SNAPSHOT.jar" in classpath

    def test_classpath_includes_forge_ai_jar(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        cp_index = cmd.index("-cp")
        classpath = cmd[cp_index + 1]
        assert "forge-ai-2.0.10-SNAPSHOT.jar" in classpath

    def test_output_file_passed_as_system_property(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "sealed" / "match-outcomes.txt"

        fake = FakeJar(tmp_path)
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(output_file)

        cmd = mock_popen.call_args[0][0]
        sys_prop = f"-Doutput.file={output_file}"
        assert sys_prop in cmd

    def test_start_returns_popen_handle(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        fake = FakeJar(tmp_path)
        mock_proc = MagicMock()
        with patch.object(connector, "_resolve_jar_path", return_value=fake.connector_jar()):
            with patch("subprocess.Popen", return_value=mock_proc):
                result = connector.start(output_file)

        assert result is mock_proc

    def test_jar_not_found_raises_file_not_found(self, tmp_path):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "outcomes.txt"

        with patch.object(connector, "_resolve_jar_path",
                          side_effect=FileNotFoundError("JAR not found")):
            with pytest.raises(FileNotFoundError):
                connector.start(output_file)
