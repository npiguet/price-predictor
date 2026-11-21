"""Unit tests for how ``train-scorer`` builds or resumes its scorer: the
inherited architecture, the optimizer's parameter groups, the resume
precedence and the embedding width checks."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.application.scorer_encoder_loading import _check_encoder_width
from sealed.application.scorer_resume import (
    _build_optimizer,
    _check_scorer_width,
    _resume_or_build_model,
)
from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase
from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.cli_train_scorer import run_train_scorer
from sealed.infrastructure.scorer_store import ScorerStore
from tests.unit.sealed.application.train_scorer_support import (
    ScorerCorpus,
    _config,
    _tiny_transformer_config,
)


def _save_phase_a_checkpoint(path: Path, *, scorer_config: ScorerConfig | None = None,
                              train_config: dict | None = None) -> None:
    cfg = scorer_config or ScorerConfig()
    model = SetTransformerScorer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ScorerStore().save_checkpoint(
        model, optimizer, epoch=1, best_val_accuracy=0.4,
        config=cfg, path=path, train_config=train_config or {},
    )


def _save_phase_b_checkpoint(
    path: Path, *, train_config: dict | None = None,
) -> tuple[TransformerConfig, CardPriceTransformerModel, ScorerConfig]:
    scorer_cfg = ScorerConfig()
    scorer = SetTransformerScorer(scorer_cfg)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    encoder_cfg = _tiny_transformer_config()
    encoder = CardPriceTransformerModel(encoder_cfg)
    from dataclasses import asdict
    ScorerStore().save_checkpoint(
        scorer, optimizer, epoch=2, best_val_accuracy=0.55,
        config=scorer_cfg, path=path,
        encoder_state_dict=encoder.state_dict(),
        encoder_config=asdict(encoder_cfg),
        train_config=train_config or {},
    )
    return encoder_cfg, encoder, scorer_cfg


class TestArchitectureInheritance:
    """T016 (US1): when --scorer-checkpoint bootstraps a fresh Phase B run, the
    constructed scorer's architecture matches the checkpoint's stored config
    and the CLI architecture-flag defaults are ignored (FR-003a)."""

    def test_scorer_checkpoint_inherits_architecture(self, tmp_path):
        ckpt_path = tmp_path / "phase_a.pt"
        custom_cfg = ScorerConfig(
            n_layers=3, n_heads=2, n_seeds=2, d_ff=64, mlp_hidden=32, dropout=0.0,
        )
        _save_phase_a_checkpoint(ckpt_path, scorer_config=custom_cfg)

        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            scorer_checkpoint=ckpt_path,
            embedding_lr=1e-7,
            # CLI architecture flags would default to 6/4/4/1088/256/0.2 here;
            # the scorer must inherit the checkpoint's smaller architecture.
        )
        resume = _resume_or_build_model(
            config, ScorerStore(), embedding_dim=custom_cfg.d_model,
        )
        assert resume.model.config.n_layers == 3
        assert resume.model.config.d_ff == 64
        assert resume.model.config.mlp_hidden == 32


class TestAdamWTwoGroupDispatch:
    """T017 (US1): _build_optimizer returns one group in Phase A, two groups
    in Phase B; AdamW in both (FR-005a, FR-005)."""

    def test_phase_a_single_group(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "o.txt",
            cards_path=tmp_path / "c",
            checkpoint_dir=tmp_path / "ckpt",
            embedding_lr=0.0,
        )
        scorer = SetTransformerScorer(ScorerConfig())
        optimizer = _build_optimizer(scorer, config)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0].get("name") == "scorer"

    def test_phase_b_two_groups_with_encoder(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "o.txt",
            cards_path=tmp_path / "c",
            checkpoint_dir=tmp_path / "ckpt",
            embedding_lr=1e-7,
            lr=1e-5,
        )
        scorer = SetTransformerScorer(ScorerConfig())
        encoder = CardPriceTransformerModel(_tiny_transformer_config())
        optimizer = _build_optimizer(scorer, config, encoder=encoder)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2
        names = [g.get("name") for g in optimizer.param_groups]
        assert names == ["scorer", "encoder"]
        scorer_group, encoder_group = optimizer.param_groups
        assert scorer_group["lr"] == 1e-5
        assert encoder_group["lr"] == 1e-7
        encoder_param_ids = {id(p) for p in encoder.parameters()}
        group_param_ids = {id(p) for p in encoder_group["params"]}
        assert group_param_ids == encoder_param_ids


def _resumed_train_config(tmp_path: Path) -> dict:
    """The training flags the resumed Phase B checkpoint carries."""
    return {
        "lr": 2e-5, "embedding_lr": 1e-7, "patience": 10,
        "outcomes_path": "missing.txt", "cards_path": "cards",
        "checkpoint_dir": str(tmp_path / "out"),
    }


def _resume_args(ckpt: Path, *, lr: float | None) -> Namespace:
    """`train-scorer --resume <ckpt>` with no other flag but ``--lr`` when given."""
    return Namespace(
        outcomes_path=None, cards_path=None, embeddings_path=None, checkpoint_dir=None,
        resume=str(ckpt), scorer_checkpoint=None, encoder_checkpoint=None,
        epochs=None, batch_size=None, lr=lr,
        n_layers=None, n_heads=None, n_seeds=None,
        d_ff=None, mlp_hidden=None, dropout=None,
        embedding_lr=None, patience=None, val_fraction=None, random_seed=None,
    )


def _config_built_by_cli(ns: Namespace) -> TrainScorerConfig:
    """The config ``run_train_scorer`` hands the use case, caught before training."""
    captured: dict = {}

    def fake_execute(self, cfg):  # noqa: ARG001
        captured["config"] = cfg
        raise SystemExit(0)

    with patch.object(TrainScorerUseCase, "execute", fake_execute):
        try:
            run_train_scorer(ns)
        except SystemExit:
            pass

    return captured["config"]


class TestResumePrecedence:
    """T048 (US1): explicit CLI > resumed train_config > dataclass default
    (FR-010, contracts/checkpoint-format.md §Resume Precedence)."""

    def test_resume_inherits_train_config_when_no_cli_flags(self, tmp_path, capsys):
        ckpt = tmp_path / "phase_b.pt"
        encoder_cfg, _, _ = _save_phase_b_checkpoint(
            ckpt, train_config=_resumed_train_config(tmp_path),
        )

        cfg = _config_built_by_cli(_resume_args(ckpt, lr=None))
        assert cfg.lr == 2e-5
        assert cfg.patience == 10
        assert cfg.embedding_lr == 1e-7

    def test_explicit_cli_flag_overrides_resumed_value(self, tmp_path):
        ckpt = tmp_path / "phase_b.pt"
        _save_phase_b_checkpoint(ckpt, train_config=_resumed_train_config(tmp_path))

        cfg = _config_built_by_cli(_resume_args(ckpt, lr=7e-6))  # explicit CLI override
        assert cfg.lr == 7e-6  # CLI wins
        assert cfg.patience == 10  # falls through to resumed value


def _tiny_sealed_encoder(d_model: int, *, pool_mode: str = "attn") -> SealedEncoderModel:
    cfg = SealedEncoderConfig(
        vocab_size=64, d_model=d_model, n_layers=1, n_heads=2, ff_dim=2 * d_model,
        max_seq_len=16, dropout=0.0, n_pool_queries=2, pool_mode=pool_mode,
    )
    return SealedEncoderModel(cfg)


class TestEmbeddingWidth:
    """The scorer's d_model is derived from the .npz cache width, not fixed at
    544; a loaded scorer / Phase-B encoder whose width disagrees fails fast."""

    def test_fresh_scorer_sized_to_embedding_cache(self, tmp_path):
        # An attn-128 encoder produces a 128 + FEATURE_COUNT wide embedding.
        embedding_dim = 128 + FEATURE_COUNT
        config = _config(
            ScorerCorpus(tmp_path / "o.txt", tmp_path / "c"), tmp_path / "ckpt", embedding_lr=0.0,
        )
        resume = _resume_or_build_model(config, ScorerStore(), embedding_dim=embedding_dim)
        assert resume.model.config.d_model == embedding_dim
        assert resume.model.det_feature_offset == 128

    def test_scorer_config_passes_d_model(self, tmp_path):
        config = _config(ScorerCorpus(tmp_path / "o.txt", tmp_path / "c"), tmp_path / "ckpt")
        assert config.scorer_config(d_model=160).d_model == 160

    def test_check_scorer_width_raises_on_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="544-wide.*is 160-wide"):
            _check_scorer_width(544, 160, "--scorer-checkpoint foo.pt", tmp_path / "c")
        # equal widths: no error
        _check_scorer_width(160, 160, "--scorer-checkpoint foo.pt", tmp_path / "c")

    def test_check_encoder_width_raises_on_mismatch(self, tmp_path):
        config = _config(ScorerCorpus(tmp_path / "o.txt", tmp_path / "c"), tmp_path / "ckpt")
        attn_encoder = _tiny_sealed_encoder(128, pool_mode="attn")  # text width 128
        # Cache says text width is 256 (≠ 128) → mismatch.
        with pytest.raises(ValueError, match="128-wide text vector.*256-wide text"):
            _check_encoder_width(attn_encoder, 256 + FEATURE_COUNT, config)
        # Matching cache: no error.
        _check_encoder_width(attn_encoder, 128 + FEATURE_COUNT, config)
        # Dual-pool sealed encoder: text width = 2*d_model.
        dual_encoder = _tiny_sealed_encoder(64, pool_mode="dual")  # text width 128
        _check_encoder_width(dual_encoder, 128 + FEATURE_COUNT, config)
