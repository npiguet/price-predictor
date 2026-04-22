"""Unit tests for MatchWorkerConnector subprocess construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab"
BEST_OF = 3


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
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "java"

    def test_command_includes_xmx_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert "-Xmx1200m" in cmd

    def test_command_includes_main_class(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert "com.pricepredictor.connector.MatchWorkerMain" in cmd

    def test_command_includes_classpath_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert "-cp" in cmd
        cp_index = cmd.index("-cp")
        assert cmd[cp_index + 1] == "fake-classpath"

    def test_output_file_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "sealed" / "match-outcomes.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(output_file, run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert f"-Doutput.file={output_file}" in cmd

    def test_start_returns_popen_handle(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc):
            result = connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        assert result is mock_proc

    def test_log_file_used_when_provided(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        log_file = MagicMock()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                log_file=log_file,
            )
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
                connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)


class TestMatchWorkerConnectorGeneratedDecksPath:
    """US3: --generated-decks-path triggers self-play mode in the Java worker."""

    def test_generated_decks_path_added_as_system_property(
        self, tmp_path, stub_classpath,
    ):
        connector = MatchWorkerConnector()
        gen_decks = tmp_path / "generated-decks.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                generated_decks_path=gen_decks,
                self_play_label="gen-2",
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dgenerated.decks.file={gen_decks}" in cmd

    def test_generated_decks_path_omitted_when_none(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                generated_decks_path=None,
            )
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dgenerated.decks.file=") for arg in cmd)

    def test_generated_decks_path_default_is_none(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dgenerated.decks.file=") for arg in cmd)


class TestMatchWorkerConnectorRunIdAndLabel:
    """run_id is required; self_play_label is required iff generated_decks_path is set."""

    def test_run_id_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert f"-Dmatch.run.id={RUN_ID}" in cmd

    def test_self_play_label_passed_as_system_property_with_gendecks(
        self, tmp_path, stub_classpath,
    ):
        connector = MatchWorkerConnector()
        gen_decks = tmp_path / "generated-decks.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                generated_decks_path=gen_decks,
                self_play_label="gen-2",
            )
        cmd = mock_popen.call_args[0][0]
        assert "-Dself.play.label=gen-2" in cmd

    def test_self_play_label_absent_without_gendecks(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF)
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dself.play.label=") for arg in cmd)

    def test_label_without_gendecks_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="forbidden"):
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                self_play_label="gen-2",
            )

    def test_gendecks_without_label_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        gen_decks = tmp_path / "generated-decks.txt"
        with pytest.raises(ValueError, match="required"):
            connector.start(
                tmp_path / "outcomes.txt",
                run_id=RUN_ID,
                best_of=BEST_OF,
                generated_decks_path=gen_decks,
            )


class TestMatchWorkerConnectorBestOf:
    """best_of plumbs through as -Dmatch.best.of and validates odd+positive."""

    def test_best_of_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=7)
        cmd = mock_popen.call_args[0][0]
        assert "-Dmatch.best.of=7" in cmd

    def test_odd_values_accepted_including_large_ones(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        for n in (1, 3, 7, 17, 101):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=n)
            cmd = mock_popen.call_args[0][0]
            assert f"-Dmatch.best.of={n}" in cmd

    def test_even_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=4)

    def test_zero_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=0)

    def test_negative_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=-3)
