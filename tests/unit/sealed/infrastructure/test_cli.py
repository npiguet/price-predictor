"""Unit tests for sealed CLI validation logic."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from sealed.infrastructure.cli import run_match_outcomes


def _args(**overrides) -> Namespace:
    """Build a minimal argparse Namespace for run_match_outcomes."""
    defaults = {
        "workers": 1,
        "generated_decks_path": None,
        "self_play_label": None,
        "best_of": 7,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestMatchOutcomesSelfPlayLabelXor:
    """--self-play-label is required iff --generated-decks-path is given."""

    def test_neither_is_valid(self, tmp_path):
        """Phase-0 mode: no self-play, no label. Should start supervisor."""
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            mock_sup.return_value.run.return_value = None
            rc = run_match_outcomes(_args())
        assert rc == 0
        mock_sup.assert_called_once()

    def test_both_is_valid(self, tmp_path):
        """Self-play mode: both given. Should start supervisor with label."""
        gen = tmp_path / "generated-decks.txt"
        gen.touch()
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            mock_sup.return_value.run.return_value = None
            rc = run_match_outcomes(_args(
                generated_decks_path=str(gen),
                self_play_label="gen-2",
            ))
        assert rc == 0
        kwargs = mock_sup.call_args.kwargs
        assert kwargs["self_play_label"] == "gen-2"
        assert kwargs["generated_decks_path"] == gen

    def test_gendecks_without_label_rejected(self, tmp_path, capsys):
        gen = tmp_path / "generated-decks.txt"
        gen.touch()
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            rc = run_match_outcomes(_args(generated_decks_path=str(gen)))
        assert rc == 2
        mock_sup.assert_not_called()
        err = capsys.readouterr().err
        assert "--self-play-label is required" in err

    def test_label_without_gendecks_rejected(self, capsys):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            rc = run_match_outcomes(_args(self_play_label="gen-2"))
        assert rc == 2
        mock_sup.assert_not_called()
        err = capsys.readouterr().err
        assert "--self-play-label is only valid" in err


class TestMatchOutcomesBestOf:
    """--best-of default is 7; value is forwarded; odd+positive is enforced."""

    def test_default_is_seven(self):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            mock_sup.return_value.run.return_value = None
            rc = run_match_outcomes(_args())
        assert rc == 0
        kwargs = mock_sup.call_args.kwargs
        assert kwargs["best_of"] == 7

    def test_odd_values_accepted_including_large(self):
        for n in (1, 3, 7, 17, 101):
            with patch(
                "sealed.application.match_outcomes.MatchOutcomeSupervisor"
            ) as mock_sup:
                mock_sup.return_value.run.return_value = None
                rc = run_match_outcomes(_args(best_of=n))
            assert rc == 0
            kwargs = mock_sup.call_args.kwargs
            assert kwargs["best_of"] == n

    def test_even_rejected(self, capsys):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            rc = run_match_outcomes(_args(best_of=4))
        assert rc == 2
        mock_sup.assert_not_called()
        err = capsys.readouterr().err
        assert "positive odd integer" in err

    def test_zero_rejected(self, capsys):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            rc = run_match_outcomes(_args(best_of=0))
        assert rc == 2
        mock_sup.assert_not_called()

    def test_negative_rejected(self, capsys):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            rc = run_match_outcomes(_args(best_of=-3))
        assert rc == 2
        mock_sup.assert_not_called()
