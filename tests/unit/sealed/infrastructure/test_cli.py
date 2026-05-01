"""Unit tests for sealed CLI validation logic."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.cli import (
    _parse_label,
    _parse_restarts,
    run_encode_cards,
    run_match_outcomes,
    run_train_scorer,
)
from sealed.infrastructure.scorer_store import ScorerStore


def _args(**overrides) -> Namespace:
    """Build a minimal argparse Namespace for run_match_outcomes."""
    defaults = {
        "workers": 1,
        "generated_decks_path": None,
        "best_of": 7,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestMatchOutcomesGeneratedDecksPath:
    """``--generated-decks-path`` is optional; supervisor receives it as-is."""

    def test_no_path_is_phase_0(self, tmp_path):
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            mock_sup.return_value.run.return_value = None
            rc = run_match_outcomes(_args())
        assert rc == 0
        kwargs = mock_sup.call_args.kwargs
        assert kwargs["generated_decks_path"] is None

    def test_explicit_path_forwarded(self, tmp_path):
        gen = tmp_path / "generated-decks.txt"
        gen.touch()
        with patch(
            "sealed.application.match_outcomes.MatchOutcomeSupervisor"
        ) as mock_sup:
            mock_sup.return_value.run.return_value = None
            rc = run_match_outcomes(_args(generated_decks_path=str(gen)))
        assert rc == 0
        kwargs = mock_sup.call_args.kwargs
        assert kwargs["generated_decks_path"] == gen


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


def _train_args(**overrides) -> Namespace:
    """Build a Namespace with every train-scorer flag at its sentinel default."""
    defaults: dict[str, object | None] = {
        "outcomes_path": None,
        "cards_path": None,
        "checkpoint_dir": None,
        "resume": None,
        "scorer_checkpoint": None,
        "encoder_checkpoint": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "n_layers": None,
        "n_heads": None,
        "n_seeds": None,
        "d_ff": None,
        "mlp_hidden": None,
        "dropout": None,
        "embedding_lr": None,
        "patience": None,
        "val_fraction": None,
        "random_seed": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_phase_a_checkpoint(path: Path) -> None:
    """Write a minimal Phase A checkpoint (no encoder fields)."""
    cfg = ScorerConfig()
    model = SetTransformerScorer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ScorerStore().save_checkpoint(
        model, optimizer, epoch=1, best_val_accuracy=0.5,
        config=cfg, path=path,
        train_config={"lr": 1e-5, "embedding_lr": 0.0},
    )


def _write_phase_b_checkpoint(path: Path) -> None:
    """Write a minimal Phase B checkpoint (with encoder fields)."""
    cfg = ScorerConfig()
    model = SetTransformerScorer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ScorerStore().save_checkpoint(
        model, optimizer, epoch=2, best_val_accuracy=0.6,
        config=cfg, path=path,
        encoder_state_dict={"layer.weight": torch.zeros(2, 3)},
        encoder_config={
            "d_model": 64, "n_layers": 1, "n_heads": 2, "ff_dim": 64,
            "max_seq_len": 16, "vocab_size": 32, "dropout": 0.0,
        },
        train_config={"lr": 2e-5, "embedding_lr": 1e-7, "patience": 10},
    )


class TestTrainScorerCliRejections:
    """T015 (US1): the run_train_scorer CLI layer rejects every invalid
    invocation shape with the contract's exact error messages."""

    def test_resume_phase_a_with_phase_b_lr_rejected(self, tmp_path, capsys):
        ckpt = tmp_path / "phase_a.pt"
        _write_phase_a_checkpoint(ckpt)
        rc = run_train_scorer(_train_args(resume=str(ckpt), embedding_lr=1e-7))
        assert rc == 2
        err = capsys.readouterr().err
        assert "is a Phase A checkpoint" in err
        assert "Phase B" in err

    def test_resume_phase_b_with_phase_a_lr_rejected(self, tmp_path, capsys):
        ckpt = tmp_path / "phase_b.pt"
        _write_phase_b_checkpoint(ckpt)
        rc = run_train_scorer(_train_args(resume=str(ckpt), embedding_lr=0.0))
        assert rc == 2
        err = capsys.readouterr().err
        assert "is a Phase B checkpoint" in err
        assert "Phase A" in err

    def test_resume_phase_b_with_explicit_encoder_checkpoint_rejected(
        self, tmp_path, capsys,
    ):
        ckpt = tmp_path / "phase_b.pt"
        _write_phase_b_checkpoint(ckpt)
        encoder = tmp_path / "encoder.pt"
        encoder.touch()
        rc = run_train_scorer(_train_args(
            resume=str(ckpt),
            embedding_lr=1e-7,
            encoder_checkpoint=str(encoder),
        ))
        assert rc == 2
        err = capsys.readouterr().err
        assert "--encoder-checkpoint conflicts with --resume" in err

    def test_resume_and_scorer_checkpoint_mutually_exclusive(
        self, tmp_path, capsys,
    ):
        a = tmp_path / "phase_a.pt"
        _write_phase_a_checkpoint(a)
        b = tmp_path / "phase_a_other.pt"
        _write_phase_a_checkpoint(b)
        rc = run_train_scorer(_train_args(
            resume=str(a),
            scorer_checkpoint=str(b),
        ))
        assert rc == 2
        err = capsys.readouterr().err
        assert "mutually exclusive" in err

    def test_scorer_checkpoint_with_architecture_flag_rejected(
        self, tmp_path, capsys,
    ):
        ckpt = tmp_path / "phase_a.pt"
        _write_phase_a_checkpoint(ckpt)
        rc = run_train_scorer(_train_args(
            scorer_checkpoint=str(ckpt),
            embedding_lr=1e-7,
            n_layers=3,
        ))
        assert rc == 2
        err = capsys.readouterr().err
        assert "--n-layers" in err
        assert "conflicts with --scorer-checkpoint" in err

    def test_bare_phase_b_lr_without_bootstrap_rejected(self, capsys):
        rc = run_train_scorer(_train_args(embedding_lr=1e-7))
        assert rc == 2
        err = capsys.readouterr().err
        assert "requires either --scorer-checkpoint" in err
        assert "or --resume" in err

    def test_scorer_checkpoint_without_embedding_lr_rejected(self, tmp_path, capsys):
        """`--scorer-checkpoint` is Phase B-only; without `--embedding-lr` the
        run would silently bootstrap and continue as Phase A, which is never
        what the user wants (`--resume` is the Phase A continuation path)."""
        ckpt = tmp_path / "phase_a.pt"
        _write_phase_a_checkpoint(ckpt)
        rc = run_train_scorer(_train_args(scorer_checkpoint=str(ckpt)))
        assert rc == 2
        err = capsys.readouterr().err
        assert "--scorer-checkpoint is only valid" in err
        assert "Phase B" in err

    def test_explicit_encoder_checkpoint_in_phase_a_rejected(self, tmp_path, capsys):
        """Explicit `--encoder-checkpoint` with no Phase B intent is a foot-gun
        — silently a no-op in Phase A. Reject it."""
        encoder = tmp_path / "encoder.pt"
        encoder.touch()
        rc = run_train_scorer(_train_args(encoder_checkpoint=str(encoder)))
        assert rc == 2
        err = capsys.readouterr().err
        assert "--encoder-checkpoint has no effect on a Phase A run" in err


def _encode_cards_args(**overrides) -> Namespace:
    """Build a Namespace for run_encode_cards with sentinel defaults."""
    defaults: dict[str, object | None] = {
        "encoder_checkpoint": None,
        "scorer_checkpoint": None,
        "vocab_path": "models/price-predictor/transformer/vocab.txt",
        "cards_path": "output/cardsfolder/",
        "clean": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestEncodeCardsMutualExclusivity:
    """T033 (US2): explicit --encoder-checkpoint + --scorer-checkpoint rejected;
    --scorer-checkpoint alone (default --encoder-checkpoint) accepted (FR-013)."""

    def test_explicit_both_rejected(self, tmp_path, capsys):
        scorer = tmp_path / "phase_b.pt"
        scorer.touch()
        encoder = tmp_path / "encoder.pt"
        encoder.touch()
        rc = run_encode_cards(_encode_cards_args(
            encoder_checkpoint=str(encoder),
            scorer_checkpoint=str(scorer),
        ))
        assert rc == 2
        err = capsys.readouterr().err
        assert "mutually exclusive" in err

    def test_scorer_checkpoint_alone_accepted_through_check(
        self, tmp_path, capsys, phase_b_setup,
    ):
        """The mutual-exclusivity check must NOT trigger when only
        --scorer-checkpoint is passed (default --encoder-checkpoint is None)."""
        from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase

        # Need a Phase B checkpoint. Run a tiny Phase B to make one.
        ckpt_dir = phase_b_setup["cards_dir"].parent / "checkpoints_b"
        config = TrainScorerConfig(
            outcomes_path=phase_b_setup["outcomes_file"],
            cards_path=phase_b_setup["cards_dir"],
            checkpoint_dir=ckpt_dir,
            scorer_checkpoint=phase_b_setup["phase_a_checkpoint"],
            encoder_checkpoint=phase_b_setup["encoder_checkpoint"],
            embedding_lr=1e-5, lr=1e-5, epochs=1, patience=10, batch_size=4,
        )
        TrainScorerUseCase().execute(config)
        phase_b = ckpt_dir / config.best_checkpoint_name()

        rc = run_encode_cards(_encode_cards_args(
            scorer_checkpoint=str(phase_b),
            vocab_path=str(phase_b_setup["vocab_path"]),
            cards_path=str(phase_b_setup["cards_dir"]),
            clean=True,
        ))
        # No mutual-exclusivity error; rc should be 0 or 1 (errors-but-completed)
        assert rc in (0, 1)


class TestEncodeCardsPhaseARejected:
    """T034 (US2): scorer-checkpoint pointed at a Phase A checkpoint rejected
    with the contract's error message (FR-014)."""

    def test_phase_a_rejected_with_helpful_message(self, tmp_path, capsys):
        ckpt = tmp_path / "phase_a.pt"
        _write_phase_a_checkpoint(ckpt)
        rc = run_encode_cards(_encode_cards_args(
            scorer_checkpoint=str(ckpt),
        ))
        assert rc == 2
        err = capsys.readouterr().err
        assert "Phase A scorer checkpoint" in err
        assert "--encoder-checkpoint" in err


class TestParseRestarts:
    """``--restarts`` accepts a positive int or the literal 'color-pairs'."""

    def test_positive_integer(self):
        assert _parse_restarts("1") == 1
        assert _parse_restarts("4") == 4
        assert _parse_restarts("100") == 100

    def test_color_pairs_literal(self):
        assert _parse_restarts("color-pairs") == "color-pairs"

    def test_zero_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
            _parse_restarts("0")

    def test_negative_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
            _parse_restarts("-3")

    def test_unknown_string_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            _parse_restarts("foo")

    def test_unknown_string_includes_value_in_message(self):
        with pytest.raises(argparse.ArgumentTypeError, match="'random-init'"):
            _parse_restarts("random-init")


class TestParseLabel:
    """``--label`` is a non-empty string with no ';', '|', or whitespace."""

    def test_alphanum_hyphen_accepted(self):
        assert _parse_label("gen-2") == "gen-2"
        assert _parse_label("gen-3-experimental") == "gen-3-experimental"
        assert _parse_label("forge-best") == "forge-best"
        assert _parse_label("a") == "a"

    def test_empty_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
            _parse_label("")

    def test_semicolon_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="';' or '\\|'"):
            _parse_label("has;semi")

    def test_pipe_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="';' or '\\|'"):
            _parse_label("has|pipe")

    def test_whitespace_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="whitespace"):
            _parse_label("has space")

    def test_tab_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="whitespace"):
            _parse_label("has\ttab")
