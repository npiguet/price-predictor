"""Unit tests for MatchWorkerConnector subprocess construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.match_worker_connector import MatchWorkerConnector


@pytest.fixture
def stub_classpath():
    """Stub build_forge_classpath at the consumer site."""
    with patch(
        "sealed.infrastructure.match_worker_connector.build_forge_classpath",
        return_value="fake-classpath",
    ):
        yield


class TestMatchWorkerConnectorCommandConstruction:
    def test_command_starts_with_java(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt")
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "java"

    def test_command_includes_xmx_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt")
        cmd = mock_popen.call_args[0][0]
        assert "-Xmx1200m" in cmd

    def test_command_includes_main_class(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt")
        cmd = mock_popen.call_args[0][0]
        assert "com.pricepredictor.connector.MatchWorkerMain" in cmd

    def test_command_includes_classpath_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt")
        cmd = mock_popen.call_args[0][0]
        assert "-cp" in cmd
        cp_index = cmd.index("-cp")
        assert cmd[cp_index + 1] == "fake-classpath"

    def test_output_file_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "sealed" / "match-outcomes.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(output_file)
        cmd = mock_popen.call_args[0][0]
        assert f"-Doutput.file={output_file}" in cmd

    def test_start_returns_popen_handle(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc):
            result = connector.start(tmp_path / "outcomes.txt")
        assert result is mock_proc

    def test_log_file_used_when_provided(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        log_file = MagicMock()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", log_file=log_file)
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["stdout"] is log_file
        assert kwargs["stderr"] is log_file

    def test_jar_not_found_raises_file_not_found(self, tmp_path):
        connector = MatchWorkerConnector()
        with patch(
            "sealed.infrastructure.match_worker_connector.build_forge_classpath",
            side_effect=FileNotFoundError("JAR not found"),
        ):
            with pytest.raises(FileNotFoundError):
                connector.start(tmp_path / "outcomes.txt")
