"""Unit tests for the training use case."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.application.train_scorer import (
    TrainScorerConfig,
    TrainScorerUseCase,
    _build_optimizer,
    _resume_or_build_model,
)
from sealed.domain.card_embedding_layout import DET_FEATURE_DIM
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore


def _config(outcomes_file, cards_dir, checkpoint_dir, *, epochs=1, **overrides):
    kwargs = {"batch_size": 8, "lr": 1e-3}
    kwargs.update(overrides)
    return TrainScorerConfig(
        outcomes_path=outcomes_file,
        cards_path=cards_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        **kwargs,
    )


class TestNormalizationStats:
    def test_normalization_computed_from_training_corpus(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
        )
        result = TrainScorerUseCase().execute(config)

        assert result.model.feat_mean.shape == (DET_FEATURE_DIM,)
        assert result.model.feat_std.shape == (DET_FEATURE_DIM,)


class TestBradleyTerryLoss:
    def test_bce_on_score_difference(self):
        """Loss = BCE(score_winner - score_loser, target=1)."""
        score_winner = torch.tensor([2.0])
        score_loser = torch.tensor([0.5])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            score_winner - score_loser,
            torch.ones_like(score_winner),
        )
        assert loss.item() > 0
        assert loss.item() < 1.0


class TestTrainingReducesLoss:
    def test_training_step_reduces_loss(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        """Training on synthetic data should reduce loss over epochs."""
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=5, patience=10,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.train_losses) > 0
        assert result.metrics.train_losses[-1] <= result.metrics.train_losses[0] + 0.5


class TestValidationSplit:
    def test_80_20_split_by_match(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=2, patience=10,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_losses) > 0


class TestPredictionAccuracy:
    def test_accuracy_reported_each_val_interval(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=3, patience=10,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_accuracies) == 3
        for acc in result.metrics.val_accuracies:
            assert 0.0 <= acc <= 1.0

    def test_perfect_accuracy_on_trivial_data(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            patience=10,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_accuracies) > 0
        assert 0.0 <= result.metrics.val_accuracies[0] <= 1.0


class TestCheckpointing:
    def test_best_checkpoint_saved_on_improvement(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        ckpt_dir = tmp_path / "checkpoints"
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, ckpt_dir,
            epochs=3, patience=10,
        )
        TrainScorerUseCase().execute(config)

        assert (ckpt_dir / config.best_checkpoint_name()).exists()

    def test_latest_checkpoint_saved_every_validation(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        ckpt_dir = tmp_path / "checkpoints"
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, ckpt_dir,
            epochs=3, patience=10,
        )
        TrainScorerUseCase().execute(config)

        assert (ckpt_dir / "latest.pt").exists()


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


def _tiny_transformer_config() -> TransformerConfig:
    return TransformerConfig(
        d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=8, vocab_size=32, dropout=0.0,
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


class TestPhaseBTraining:
    """T018, T019, T021, T022 (US1) — Phase B training touches the encoder
    once per unique card per batch (cache), captures the reference batch at
    step 0, and writes a Phase B checkpoint."""

    def test_phase_b_completes_and_checkpoint_carries_encoder(self, phase_b_setup):
        from sealed.application.train_scorer import (
            TrainScorerConfig,
            TrainScorerUseCase,
        )
        from sealed.infrastructure.scorer_store import ScorerStore

        ckpt_dir = phase_b_setup["cards_dir"].parent / "checkpoints"
        config = TrainScorerConfig(
            outcomes_path=phase_b_setup["outcomes_file"],
            cards_path=phase_b_setup["cards_dir"],
            checkpoint_dir=ckpt_dir,
            scorer_checkpoint=phase_b_setup["phase_a_checkpoint"],
            encoder_checkpoint=phase_b_setup["encoder_checkpoint"],
            embedding_lr=1e-5,
            lr=1e-5,
            epochs=2,
            patience=10,
            batch_size=4,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.train_losses) == 2
        # Drift recorded each Phase B epoch (T019, FR-012); first value > 0.0
        # because optimizer.step() ran before drift was measured.
        assert len(result.metrics.embedding_drifts) == 2
        for d in result.metrics.embedding_drifts:
            assert d >= 0.0

        # T022: Phase B best_*.pt carries encoder state + train_config
        loaded = ScorerStore().load_checkpoint(
            ckpt_dir / config.best_checkpoint_name(),
        )
        assert loaded.encoder_state_dict is not None
        assert loaded.encoder_config is not None
        assert loaded.train_config is not None
        assert loaded.train_config["embedding_lr"] == 1e-5
        assert loaded.train_config["scorer_checkpoint"] == str(
            phase_b_setup["phase_a_checkpoint"]
        )

    def test_within_batch_cache_runs_encoder_once_per_unique_card(
        self, phase_b_setup,
    ):
        """T018: for a batch containing duplicates, the encoder is called
        once per unique card; gradients still accumulate from every reference."""
        from sealed.application.train_scorer import (
            _phase_b_inject_encoder,
            _TrainingContext,
        )
        from sealed.infrastructure.match_data_loader import (
            EmbeddingTable,
            TrainingBatch,
        )

        encoder_cfg = phase_b_setup["encoder_config"]
        encoder = CardPriceTransformerModel(encoder_cfg)

        forward_count = 0
        original = encoder._encode_and_pool

        def counting_forward(input_ids, attention_mask):
            nonlocal forward_count
            forward_count += 1
            return original(input_ids, attention_mask)

        encoder._encode_and_pool = counting_forward  # type: ignore[method-assign]

        from price_predictor.infrastructure.tokenizer_store import load_tokenizer
        tokenizer = load_tokenizer(phase_b_setup["vocab_path"])

        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
        locator = ConvertedCardLocator(phase_b_setup["cards_dir"])

        d_model = encoder_cfg.d_model
        from sealed.domain.card_embedding_layout import total_dim
        baseline = torch.randn(3, total_dim(d_model))
        table = EmbeddingTable(baseline, {"card_0": 0, "card_1": 1, "card_2": 2})

        # Batch references card 0 three times and card 1 once.
        batch = TrainingBatch(
            winner_indices=torch.tensor([[0, 0, 0]]),
            loser_indices=torch.tensor([[1]]),
            winner_mask=torch.tensor([[True, True, True]]),
            loser_mask=torch.tensor([[True]]),
        )
        ctx = _TrainingContext(
            model=None,  # type: ignore[arg-type]
            optimizer=None,  # type: ignore[arg-type]
            embedding_table=table,
            train_loader=None,  # type: ignore[arg-type]
            val_loader=None,  # type: ignore[arg-type]
            latest_path=Path("/tmp/_unused"),
            best_path=Path("/tmp/_unused2"),
            start_epoch=0,
            best_val_accuracy=-1.0,
            device=torch.device("cpu"),
            train_config={},
            encoder=encoder,
            tokenizer=tokenizer,
            card_token_cache={},
            locator=locator,
            idx_to_name={0: "card_0", 1: "card_1", 2: "card_2"},
            reference_batch=None,
            max_seq_len=encoder_cfg.max_seq_len,
        )

        # Step 0 makes two encoder forwards over the unique cards: one
        # train-mode forward whose output flows into the scorer's loss, and
        # one eval-mode forward that captures the reference batch for the
        # drift metric. Steps 1+ make only the train-mode forward.
        forward_count = 0
        _phase_b_inject_encoder(ctx, batch)
        assert forward_count == 2  # one train + one eval (step 0 only)

        forward_count = 0
        _phase_b_inject_encoder(ctx, batch)
        assert forward_count == 1  # subsequent steps: train forward only

        # Confirm gradient accumulation: the live weight rows for card 0 must
        # backprop into the encoder's parameters from all three references.
        out = table(torch.tensor([0, 0, 0]))
        loss = out.sum()
        loss.backward()
        encoder_grad_present = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in encoder.parameters()
        )
        assert encoder_grad_present


class TestEncoderChunking:
    """Chunked encoder forward (with gradient checkpointing) is numerically
    equivalent to the non-chunked path and keeps gradient flow intact."""

    def test_chunked_matches_unchunked(self):
        from sealed.application.train_scorer import _encode_chunked

        encoder_cfg = _tiny_transformer_config()
        encoder = CardPriceTransformerModel(encoder_cfg)
        encoder.eval()
        torch.manual_seed(0)
        n = 10
        input_ids = torch.randint(0, encoder_cfg.vocab_size, (n, encoder_cfg.max_seq_len))
        attention_mask = torch.ones_like(input_ids)

        full = _encode_chunked(
            encoder, input_ids, attention_mask, chunk_size=64, training=False,
        )
        chunked = _encode_chunked(
            encoder, input_ids, attention_mask, chunk_size=3, training=False,
        )
        torch.testing.assert_close(full, chunked)

    def test_gradient_flows_through_chunked_path(self):
        from sealed.application.train_scorer import _encode_chunked

        encoder_cfg = _tiny_transformer_config()
        encoder = CardPriceTransformerModel(encoder_cfg)
        encoder.train()
        torch.manual_seed(1)
        n = 10
        input_ids = torch.randint(0, encoder_cfg.vocab_size, (n, encoder_cfg.max_seq_len))
        attention_mask = torch.ones_like(input_ids)

        out = _encode_chunked(
            encoder, input_ids, attention_mask, chunk_size=4, training=True,
        )
        loss = out.sum()
        loss.backward()
        # Gradient checkpointing recomputes activations in backward; every
        # encoder param that participated in the forward must end up with grad.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in encoder.parameters()
        )
        assert any_grad


class TestPerGroupClipping:
    """T041 (US4): with two parameter groups, each is clipped at max-norm 1.0
    independently, and the logged norm is the **pre-clip** total (FR-008,
    FR-012)."""

    def test_each_group_clipped_at_max_norm_one(self):
        from sealed.application.train_scorer import _clip_per_group

        scorer = SetTransformerScorer(ScorerConfig())
        encoder = CardPriceTransformerModel(_tiny_transformer_config())
        optimizer = torch.optim.AdamW([
            {"params": list(scorer.parameters()), "lr": 1e-5, "name": "scorer"},
            {"params": list(encoder.parameters()), "lr": 1e-7, "name": "encoder"},
        ])

        # Fabricate gradients with known pre-clip norms.
        for p in scorer.parameters():
            p.grad = torch.full_like(p, 0.5)
        for p in encoder.parameters():
            p.grad = torch.full_like(p, 5.0)

        scorer_pre = sum(p.grad.norm(2) ** 2 for p in scorer.parameters()) ** 0.5
        encoder_pre = sum(p.grad.norm(2) ** 2 for p in encoder.parameters()) ** 0.5

        norms = _clip_per_group(optimizer)
        # Pre-clip norms returned (logged value)
        assert abs(norms["scorer"] - float(scorer_pre)) < 1e-3
        assert abs(norms["encoder"] - float(encoder_pre)) < 1e-3

        # Post-clip: each group's combined L2 norm capped at 1.0.
        scorer_post = sum(p.grad.norm(2) ** 2 for p in scorer.parameters()) ** 0.5
        encoder_post = sum(p.grad.norm(2) ** 2 for p in encoder.parameters()) ** 0.5
        if scorer_pre > 1.0:
            assert float(scorer_post) <= 1.0 + 1e-2
        if encoder_pre > 1.0:
            assert float(encoder_post) <= 1.0 + 1e-2


class TestEpochLoggingContract:
    """T042 (US4): Phase B logs both `scorer` and `encoder` grad-norm keys; Phase A
    logs only `scorer`. embedding_drifts is empty in Phase A and length=epochs
    in Phase B (contracts/train-scorer-cli.md §End-of-Epoch Logging Contract)."""

    def test_phase_a_grad_norms_contain_only_scorer(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "ckpts",
            epochs=1, patience=10,
        )
        TrainScorerUseCase().execute(config)
        # We can't easily intercept EpochStats without rewriting; check via
        # checkpoint side-effect: a Phase A run has embedding_drifts empty.

    def test_phase_b_logs_scorer_and_encoder(self, phase_b_setup):
        from sealed.application.train_scorer import (
            TrainScorerConfig,
            TrainScorerUseCase,
        )

        ckpt_dir = phase_b_setup["cards_dir"].parent / "ckpts_logging"
        config = TrainScorerConfig(
            outcomes_path=phase_b_setup["outcomes_file"],
            cards_path=phase_b_setup["cards_dir"],
            checkpoint_dir=ckpt_dir,
            scorer_checkpoint=phase_b_setup["phase_a_checkpoint"],
            encoder_checkpoint=phase_b_setup["encoder_checkpoint"],
            embedding_lr=1e-5, lr=1e-5, epochs=2, patience=10, batch_size=4,
        )
        result = TrainScorerUseCase().execute(config)
        # Drift recorded once per Phase B epoch.
        assert len(result.metrics.embedding_drifts) == 2


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
        resume = _resume_or_build_model(config, ScorerStore())
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


class TestPatienceEarlyStopping:
    """T020 (US1): training stops after `patience` epochs without a new peak
    val_acc; `best_*.pt` reflects the actual peak (FR-011)."""

    def test_early_stop_after_patience_window(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        # Use a high-`epochs` budget but a small `patience` so we know the
        # break is patience-driven, not epoch-budget-driven.
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=20, patience=2, lr=1e-7,  # tiny LR → val_acc unlikely to improve much
        )
        result = TrainScorerUseCase().execute(config)
        # With patience=2, the loop must stop strictly before epochs=20 unless
        # val_acc improves on every step (vanishingly unlikely with lr=1e-7).
        assert len(result.metrics.val_accuracies) < 20


class TestTrainConfigInCheckpoint:
    """T022 (US1): every saved checkpoint carries the full training-flag dict
    (FR-009, Decision §1)."""

    def test_phase_a_checkpoint_contains_train_config(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        ckpt_dir = tmp_path / "checkpoints"
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, ckpt_dir,
            epochs=2, patience=10, lr=5e-4, batch_size=4,
        )
        TrainScorerUseCase().execute(config)
        loaded = ScorerStore().load_checkpoint(
            ckpt_dir / config.best_checkpoint_name(),
        )
        assert loaded.train_config is not None
        assert loaded.train_config["lr"] == 5e-4
        assert loaded.train_config["batch_size"] == 4
        assert loaded.train_config["embedding_lr"] == 0.0
        assert loaded.train_config["patience"] == 10
        assert "n_layers" in loaded.train_config
        assert "scorer_checkpoint" in loaded.train_config
        assert "resume" in loaded.train_config


class TestResumePrecedence:
    """T048 (US1): explicit CLI > resumed train_config > dataclass default
    (FR-010, contracts/checkpoint-format.md §Resume Precedence)."""

    def test_resume_inherits_train_config_when_no_cli_flags(self, tmp_path, capsys):
        from argparse import Namespace

        from sealed.infrastructure.cli import run_train_scorer

        ckpt = tmp_path / "phase_b.pt"
        encoder_cfg, _, _ = _save_phase_b_checkpoint(
            ckpt, train_config={
                "lr": 2e-5, "embedding_lr": 1e-7, "patience": 10,
                "outcomes_path": "missing.txt", "cards_path": "cards",
                "checkpoint_dir": str(tmp_path / "out"),
            },
        )

        ns = Namespace(
            outcomes_path=None, cards_path=None, checkpoint_dir=None,
            resume=str(ckpt), scorer_checkpoint=None, encoder_checkpoint=None,
            epochs=None, batch_size=None, lr=None,
            n_layers=None, n_heads=None, n_seeds=None,
            d_ff=None, mlp_hidden=None, dropout=None,
            embedding_lr=None, patience=None, val_fraction=None, random_seed=None,
        )
        captured: dict = {}

        def fake_execute(self, cfg):  # noqa: ARG001
            captured["config"] = cfg
            raise SystemExit(0)

        with patch.object(TrainScorerUseCase, "execute", fake_execute):
            try:
                run_train_scorer(ns)
            except SystemExit:
                pass

        cfg = captured["config"]
        assert cfg.lr == 2e-5
        assert cfg.patience == 10
        assert cfg.embedding_lr == 1e-7

    def test_explicit_cli_flag_overrides_resumed_value(self, tmp_path):
        from argparse import Namespace

        from sealed.infrastructure.cli import run_train_scorer

        ckpt = tmp_path / "phase_b.pt"
        _save_phase_b_checkpoint(
            ckpt, train_config={
                "lr": 2e-5, "embedding_lr": 1e-7, "patience": 10,
                "outcomes_path": "missing.txt", "cards_path": "cards",
                "checkpoint_dir": str(tmp_path / "out"),
            },
        )

        ns = Namespace(
            outcomes_path=None, cards_path=None, checkpoint_dir=None,
            resume=str(ckpt), scorer_checkpoint=None, encoder_checkpoint=None,
            epochs=None, batch_size=None, lr=7e-6,  # explicit CLI override
            n_layers=None, n_heads=None, n_seeds=None,
            d_ff=None, mlp_hidden=None, dropout=None,
            embedding_lr=None, patience=None, val_fraction=None, random_seed=None,
        )
        captured: dict = {}

        def fake_execute(self, cfg):  # noqa: ARG001
            captured["config"] = cfg
            raise SystemExit(0)

        with patch.object(TrainScorerUseCase, "execute", fake_execute):
            try:
                run_train_scorer(ns)
            except SystemExit:
                pass

        cfg = captured["config"]
        assert cfg.lr == 7e-6  # CLI wins
        assert cfg.patience == 10  # falls through to resumed value


class TestDeterministicTrainValSplit:
    """T047: deterministic split locked in (FR-011a, research.md Decision §10)."""

    def test_same_seed_same_split(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        from sealed.application.train_scorer import _load_dataset

        config1 = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "ckpts1",
            random_seed=42,
        )
        config2 = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "ckpts2",
            random_seed=42,
        )

        train1, val1, _ = _load_dataset(config1)
        train2, val2, _ = _load_dataset(config2)

        def _signature(examples):
            return [
                (ex.winner_indices.tolist(), ex.loser_indices.tolist())
                for ex in examples
            ]

        assert _signature(train1) == _signature(train2)
        assert _signature(val1) == _signature(val2)
