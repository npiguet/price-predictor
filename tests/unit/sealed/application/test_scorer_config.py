"""Unit tests for ``TrainScorerConfig``: its checkpoint file names and its phase."""

from __future__ import annotations

from sealed.application.train_scorer import TrainScorerConfig


class TestBestCheckpointName:
    def test_uses_arch_hyperparameters(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            n_layers=3,
            n_heads=8,
            n_seeds=2,
            d_ff=2048,
            mlp_hidden=128,
            lr=5e-4,
        )
        assert config.best_checkpoint_name() == "best_l3_h8_s2_ff2048_mlp128_lr0.0005.pt"

    def test_phase_b_filename_distinct_from_phase_a(self, tmp_path):
        """Phase B `best_*.pt` and `latest.pt` MUST NOT collide with the
        Phase A checkpoints in the same directory; the Phase A files survive
        a Phase B run unmodified so a regression can be reverted by simply
        switching back to them (SC-003)."""
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            embedding_lr=1e-7,
            lr=1e-5,
        )
        assert config.phase == "B"
        assert "phaseB" in config.best_checkpoint_name()
        assert "emblr1e-07" in config.best_checkpoint_name()
        assert config.latest_checkpoint_name() == "latest_phaseB.pt"
        # And it must not match the Phase A file the user typically
        # bootstrapped from.
        phase_a_config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            lr=1e-5,
        )
        assert (
            config.best_checkpoint_name() != phase_a_config.best_checkpoint_name()
        )
        assert (
            config.latest_checkpoint_name() != phase_a_config.latest_checkpoint_name()
        )

    def test_margin_weighting_suffix(self, tmp_path):
        """`--margin-weighting linear|log` appends a suffix so weighted runs
        do not clobber unweighted ones at the same architecture; `none`
        preserves the existing filename for backwards compatibility."""
        base_kwargs = dict(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            lr=1e-5,
        )
        unweighted = TrainScorerConfig(**base_kwargs)
        linear = TrainScorerConfig(**base_kwargs, margin_weighting="linear")
        log = TrainScorerConfig(**base_kwargs, margin_weighting="log")

        assert "_mw" not in unweighted.best_checkpoint_name()
        assert unweighted.latest_checkpoint_name() == "latest.pt"
        assert linear.best_checkpoint_name().endswith("_mwlin.pt")
        assert linear.latest_checkpoint_name() == "latest_mwlin.pt"
        assert log.best_checkpoint_name().endswith("_mwlog.pt")
        assert log.latest_checkpoint_name() == "latest_mwlog.pt"


class TestPhaseProperty:
    def test_phase_a_when_embedding_lr_zero(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        assert config.embedding_lr == 0.0
        assert config.phase == "A"

    def test_phase_b_when_embedding_lr_nonzero(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            embedding_lr=1e-7,
        )
        assert config.phase == "B"
