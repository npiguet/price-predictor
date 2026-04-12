"""Unit tests for the evaluation connector (worker command, deck builder command)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sealed.infrastructure.evaluation_connector import EvaluationConnector


class TestWorkerCommandConstruction:
    def test_builds_java_command(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(
            matches_file=Path("/tmp/matches-0.txt"),
        )
        assert "java" in cmd[0]
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd
        assert any("/tmp/matches-0.txt" in str(arg) or "\\tmp\\matches-0.txt" in str(arg) for arg in cmd)

    def test_correct_main_class(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"))
        assert "com.pricepredictor.connector.ValidationWorkerMain" in cmd

    def test_best_of_included_in_command(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"), best_of=5)
        assert "-Dbest.of=5" in cmd

    def test_default_best_of_is_three(self):
        connector = EvaluationConnector()
        cmd = connector._build_worker_command(matches_file=Path("/tmp/test.txt"))
        assert "-Dbest.of=3" in cmd


class TestDeckBuilderCommand:
    def test_correct_main_class(self):
        connector = EvaluationConnector()
        cmd = connector._build_deck_builder_command()
        assert "com.pricepredictor.connector.DeckBuilderMain" in cmd

    def test_is_java_command(self):
        connector = EvaluationConnector()
        cmd = connector._build_deck_builder_command()
        assert "java" in cmd[0]

    def test_does_not_include_matches_file_arg(self):
        connector = EvaluationConnector()
        cmd = connector._build_deck_builder_command()
        assert not any("matches.file" in str(arg) for arg in cmd)


class TestOutcomeFilePath:
    def test_derives_outcome_path(self):
        connector = EvaluationConnector()
        matches = Path("/tmp/validation-matches-0.txt")
        outcome = connector.outcome_file_path(matches)
        assert str(outcome).endswith("-outcomes.txt")
        assert "validation-matches-0.txt" in str(outcome)


class TestBuildForgeDecks:
    def test_parses_stdout_into_decks(self):
        """build_forge_decks splits stdout lines into lists of card names."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "CardA|CardB|CardC\nCardD|CardE|CardF\n"
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            decks = connector.build_forge_decks([["CardA", "CardB"], ["CardD", "CardE"]])

        assert len(decks) == 2
        assert decks[0] == ["CardA", "CardB", "CardC"]
        assert decks[1] == ["CardD", "CardE", "CardF"]

    def test_raises_on_nonzero_exit(self):
        """build_forge_decks raises RuntimeError if DeckBuilderMain fails."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "fatal error"

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="DeckBuilderMain failed"):
                connector.build_forge_decks([["CardA"]])

    def test_stdin_is_pipe_separated_pools(self):
        """build_forge_decks sends each pool as a pipe-separated line on stdin."""
        connector = EvaluationConnector()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "X|Y\n"
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            connector.build_forge_decks([["Card A", "Card B"]])

        call_kwargs = mock_run.call_args
        stdin_text = call_kwargs.kwargs.get("input") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        if stdin_text is None:
            stdin_text = call_kwargs.kwargs["input"]
        assert "Card A|Card B" in stdin_text
