"""Unit tests for the evaluation connector (worker command, deck builder command)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.evaluation_connector import EvaluationConnector


@pytest.fixture
def stub_classpath():
    """Stub build_forge_classpath at the consumer site."""
    with patch(
        "sealed.infrastructure.evaluation_connector.build_forge_classpath",
        return_value="fake-classpath",
    ):
        yield


class TestWorkerCommandConstruction:
    def test_builds_java_command(self, stub_classpath):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/matches-0.txt"))
        assert cmd[0] == "java"
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd
        assert any("matches-0.txt" in str(arg) for arg in cmd)

    def test_correct_main_class(self, stub_classpath):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"))
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd

    def test_best_of_included_in_command(self, stub_classpath):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"), best_of=5)
        assert "-Dbest.of=5" in cmd

    def test_default_best_of_is_three(self, stub_classpath):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"))
        assert "-Dbest.of=3" in cmd


class TestDeckBuilderCommand:
    def test_calls_run_forge_worker_with_correct_main_class(self, stub_classpath):
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "X|Y\n"
        fake_result.stderr = ""
        with patch(
            "sealed.infrastructure.evaluation_connector.run_forge_worker",
            return_value=fake_result,
        ) as mock:
            connector.build_forge_decks([["Card"]])
        mock.assert_called_once()
        assert mock.call_args[0][0] == (
            "com.pricepredictor.connector.DeckBuilderMain"
        )


class TestOutcomeFilePath:
    def test_derives_outcome_path(self):
        connector = EvaluationConnector()
        matches = Path("/tmp/validation-matches-0.txt")
        outcome = connector.outcome_file_path(matches)
        assert str(outcome).endswith("-outcomes.txt")
        assert "validation-matches-0.txt" in str(outcome)


class TestBuildForgeDecks:
    def test_parses_stdout_into_decks(self, stub_classpath):
        """build_forge_decks splits stdout lines into lists of card names."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "CardA|CardB|CardC\nCardD|CardE|CardF\n"
        fake_result.stderr = ""

        with patch(
            "sealed.infrastructure.evaluation_connector.run_forge_worker",
            return_value=fake_result,
        ):
            decks = connector.build_forge_decks([["CardA", "CardB"], ["CardD", "CardE"]])

        assert len(decks) == 2
        assert decks[0] == ["CardA", "CardB", "CardC"]
        assert decks[1] == ["CardD", "CardE", "CardF"]

    def test_raises_on_nonzero_exit(self, stub_classpath):
        """build_forge_decks raises RuntimeError if DeckBuilderMain fails."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "fatal error"

        with patch(
            "sealed.infrastructure.evaluation_connector.run_forge_worker",
            return_value=fake_result,
        ):
            with pytest.raises(RuntimeError, match="DeckBuilderMain failed"):
                connector.build_forge_decks([["CardA"]])

    def test_stdin_is_pipe_separated_pools(self, stub_classpath):
        """build_forge_decks sends each pool as a pipe-separated line on stdin."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "X|Y\n"
        fake_result.stderr = ""

        with patch(
            "sealed.infrastructure.evaluation_connector.run_forge_worker",
            return_value=fake_result,
        ) as mock_run:
            connector.build_forge_decks([["Card A", "Card B"]])

        stdin_text = mock_run.call_args.kwargs["input_text"]
        assert "Card A|Card B" in stdin_text
