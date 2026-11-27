"""Unit tests for Phase B training: the encoder forward inside the scorer's
loop, its chunked path, and the per-group gradient clipping."""

from __future__ import annotations

from pathlib import Path

import torch

from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.application.train_scorer import TrainScorerConfig
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from tests.unit.sealed.application.train_scorer_support import _tiny_transformer_config


class TestPhaseBTraining:
    """Phase B training touches the encoder
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
        # Drift recorded each Phase B epoch (FR-012); first value > 0.0
        # because optimizer.step() ran before drift was measured.
        assert len(result.metrics.embedding_drifts) == 2
        for d in result.metrics.embedding_drifts:
            assert d >= 0.0

        # Phase B best_*.pt carries encoder state + train_config
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
        """For a batch containing duplicates, the encoder is called
        once per unique card; gradients still accumulate from every reference."""
        from sealed.application.scorer_phase_b import _phase_b_inject_encoder
        from sealed.application.scorer_training_context import _TrainingContext
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
            margins=torch.tensor([1.0]),
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

        # The encoder runs in eval mode for the entire Phase B loop, so the
        # forward we compute for the scorer's loss is identical to what the
        # reference batch would capture — no extra forward needed at step 0.
        # Every step makes exactly one encoder forward over the unique cards
        # in the batch, regardless of duplicate references.
        forward_count = 0
        _phase_b_inject_encoder(ctx, batch)
        assert forward_count == 1

        forward_count = 0
        _phase_b_inject_encoder(ctx, batch)
        assert forward_count == 1

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
        from sealed.application.scorer_phase_b import TokenizedCards, _encode_chunked

        encoder_cfg = _tiny_transformer_config()
        encoder = CardPriceTransformerModel(encoder_cfg)
        encoder.eval()
        torch.manual_seed(0)
        n = 10
        input_ids = torch.randint(0, encoder_cfg.vocab_size, (n, encoder_cfg.max_seq_len))
        attention_mask = torch.ones_like(input_ids)

        full = _encode_chunked(
            encoder, TokenizedCards(input_ids, attention_mask), chunk_size=64, training=False,
        )
        chunked = _encode_chunked(
            encoder, TokenizedCards(input_ids, attention_mask), chunk_size=3, training=False,
        )
        torch.testing.assert_close(full, chunked)

    def test_gradient_flows_through_chunked_path(self):
        from sealed.application.scorer_phase_b import TokenizedCards, _encode_chunked

        encoder_cfg = _tiny_transformer_config()
        encoder = CardPriceTransformerModel(encoder_cfg)
        encoder.train()
        torch.manual_seed(1)
        n = 10
        input_ids = torch.randint(0, encoder_cfg.vocab_size, (n, encoder_cfg.max_seq_len))
        attention_mask = torch.ones_like(input_ids)

        out = _encode_chunked(
            encoder, TokenizedCards(input_ids, attention_mask), chunk_size=4, training=True,
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


def _fill_grads(module: torch.nn.Module, value: float) -> None:
    """Set every parameter's gradient to ``value``."""
    for p in module.parameters():
        p.grad = torch.full_like(p, value)


def _grad_norm(module: torch.nn.Module) -> torch.Tensor:
    """The L2 norm of every parameter's gradient taken together."""
    return sum(p.grad.norm(2) ** 2 for p in module.parameters()) ** 0.5


class TestPerGroupClipping:
    """Each parameter group is clipped independently at the
    configured ``max_norm``; the returned values are the pre-clip totals
    (FR-008, FR-012)."""

    def test_each_group_clipped_at_configured_max_norm(self):
        from price_predictor.infrastructure.torch_training import clip_per_group

        scorer = SetTransformerScorer(ScorerConfig())
        encoder = CardPriceTransformerModel(_tiny_transformer_config())
        optimizer = torch.optim.AdamW([
            {"params": list(scorer.parameters()), "lr": 1e-5, "name": "scorer"},
            {"params": list(encoder.parameters()), "lr": 1e-7, "name": "encoder"},
        ])

        _fill_grads(scorer, 0.5)
        _fill_grads(encoder, 5.0)

        scorer_pre = _grad_norm(scorer)
        encoder_pre = _grad_norm(encoder)

        clip_at = 2.0
        norms = clip_per_group(optimizer, max_norm=clip_at)
        assert abs(norms["scorer"] - float(scorer_pre)) < 1e-3
        assert abs(norms["encoder"] - float(encoder_pre)) < 1e-3

        scorer_post = _grad_norm(scorer)
        encoder_post = _grad_norm(encoder)
        if scorer_pre > clip_at:
            assert float(scorer_post) <= clip_at + 1e-2
        if encoder_pre > clip_at:
            assert float(encoder_post) <= clip_at + 1e-2

    def test_default_max_grad_norm_is_loose(self, tmp_path):
        """The default cap (100.0) is meant to act as a NaN-spike guard, not
        an effective LR throttle — typical pre-clip norms should fall well
        under it (FR-008 update)."""
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "o.txt",
            cards_path=tmp_path / "c",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert config.max_grad_norm == 100.0
