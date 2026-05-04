"""Argparse-level tests for the flipped ``train-scorer`` default (FR-024)
and the missing-default-file guard (FR-026)."""

from __future__ import annotations

from pathlib import Path

import torch

from sealed.application.train_scorer import TrainScorerConfig
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore


def _save_minimal_phase_a_checkpoint(path: Path) -> None:
    """Persist a no-op Phase A scorer checkpoint at ``path``.

    Used by tests that need the CLI to accept ``--scorer-checkpoint`` for
    Phase B kickoffs without running training.
    """
    config = ScorerConfig(n_layers=1, n_heads=2, n_seeds=2, d_ff=16, mlp_hidden=8)
    model = SetTransformerScorer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path.parent.mkdir(parents=True, exist_ok=True)
    ScorerStore().save_checkpoint(
        model, optimizer, epoch=0, best_val_accuracy=0.0,
        config=config, path=path,
    )


class TestTrainScorerDefaults:
    def test_default_encoder_checkpoint_is_sealed(self):
        config = TrainScorerConfig(
            outcomes_path=Path("x"),
            cards_path=Path("y"),
            checkpoint_dir=Path("z"),
        )
        assert config.encoder_checkpoint == Path("models/sealed/encoder/latest.pt")


class TestMissingDefaultFileGuard:
    """run_train_scorer should error when Phase B fresh kickoff would
    silently load a non-existent default sealed encoder."""

    def test_phase_b_fresh_kickoff_missing_encoder_exits_2(
        self, tmp_path: Path, capsys, monkeypatch,
    ):
        from sealed.infrastructure.cli import build_parser, run_train_scorer

        monkeypatch.chdir(tmp_path)
        scorer_ckpt = tmp_path / "scorer.pt"
        _save_minimal_phase_a_checkpoint(scorer_ckpt)

        parser = build_parser()
        args = parser.parse_args([
            "train-scorer",
            "--scorer-checkpoint", str(scorer_ckpt),
            "--embedding-lr", "1e-5",
        ])
        rc = run_train_scorer(args)
        captured = capsys.readouterr()
        assert rc == 2, captured.err
        assert "Sealed encoder not found" in captured.err
        assert "python -m sealed train-encoder" in captured.err

    def test_explicit_encoder_path_skips_guard(
        self, tmp_path: Path, capsys, monkeypatch,
    ):
        """Even when the explicit path is missing, the guard does not fire —
        the existing 'encoder model not found' error path takes over."""
        from sealed.infrastructure.cli import build_parser, run_train_scorer

        monkeypatch.chdir(tmp_path)
        scorer_ckpt = tmp_path / "scorer.pt"
        _save_minimal_phase_a_checkpoint(scorer_ckpt)
        explicit_missing = tmp_path / "nope.pt"

        parser = build_parser()
        args = parser.parse_args([
            "train-scorer",
            "--scorer-checkpoint", str(scorer_ckpt),
            "--embedding-lr", "1e-5",
            "--encoder-checkpoint", str(explicit_missing),
        ])
        rc = run_train_scorer(args)
        captured = capsys.readouterr()
        # Should NOT carry the "Sealed encoder not found" message because
        # the user opted in explicitly. Whatever error path runs (likely
        # FileNotFoundError downstream) must surface a non-zero exit code.
        assert rc != 0
        assert "Sealed encoder not found" not in captured.err

    def test_phase_a_run_does_not_fire_guard(
        self, tmp_path: Path, capsys, monkeypatch,
    ):
        """Phase A doesn't load the encoder — no guard should fire even
        when the default sealed encoder is missing."""
        from sealed.infrastructure.cli import build_parser, run_train_scorer

        monkeypatch.chdir(tmp_path)
        # Phase A run with no --embedding-lr (defaults to 0). We don't
        # need a working outcomes file — we only care that the guard
        # message does NOT appear before the inevitable downstream error.
        parser = build_parser()
        args = parser.parse_args(["train-scorer"])
        rc = run_train_scorer(args)
        captured = capsys.readouterr()
        assert "Sealed encoder not found" not in captured.err
        # Returns non-zero (no real outcomes file), but for a reason
        # unrelated to the missing default sealed encoder.
        assert rc != 0


class TestResumeUnaffected:
    def test_phase_b_resume_does_not_require_default_encoder(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        """Phase B resume pulls encoder weights from the resumed checkpoint;
        the default-flip must not break that path."""
        from sealed.infrastructure.cli import build_parser, run_train_scorer

        monkeypatch.chdir(tmp_path)
        # Build a Phase B-shaped checkpoint by attaching encoder state.
        scorer_config = ScorerConfig(
            n_layers=1, n_heads=2, n_seeds=2, d_ff=16, mlp_hidden=8,
        )
        model = SetTransformerScorer(scorer_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt_path = tmp_path / "phaseB.pt"
        ScorerStore().save_checkpoint(
            model, optimizer, epoch=0, best_val_accuracy=0.0,
            config=scorer_config, path=ckpt_path,
            encoder_state_dict={"dummy": torch.zeros(1)},
            encoder_config={
                "d_model": 8, "n_layers": 1, "n_heads": 1, "ff_dim": 8,
                "max_seq_len": 8, "vocab_size": 8, "dropout": 0.0,
            },
            train_config={"encoder_checkpoint": "models/sealed/encoder/latest.pt"},
        )
        parser = build_parser()
        args = parser.parse_args([
            "train-scorer",
            "--resume", str(ckpt_path),
            "--embedding-lr", "1e-5",
        ])
        rc = run_train_scorer(args)
        captured = capsys.readouterr()
        # The "Sealed encoder not found" guard MUST NOT fire on resume.
        assert "Sealed encoder not found" not in captured.err
        # Some downstream error is fine (we didn't seed an outcomes file).
        assert rc != 0
