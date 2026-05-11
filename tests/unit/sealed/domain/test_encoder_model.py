"""Unit tests for the sealed encoder model."""

from __future__ import annotations

import pytest
import torch

from sealed.domain.encoder_model import (
    COLOR_ORDER,
    SealedEncoderConfig,
    SealedEncoderModel,
)


def _config(**overrides) -> SealedEncoderConfig:
    base = dict(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_dim=64,
        max_seq_len=16,
        dropout=0.1,
        n_pool_queries=4,
    )
    base.update(overrides)
    return SealedEncoderConfig(**base)


def _ids_and_mask(batch: int, seq_len: int, vocab_size: int):
    input_ids = torch.randint(low=4, high=vocab_size, size=(batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return input_ids, attention_mask


class TestSealedEncoderConfig:
    def test_d_model_must_divide_n_pool_queries(self):
        with pytest.raises(ValueError, match="n_pool_queries"):
            _config(d_model=32, n_pool_queries=5)

    def test_n_heads_must_divide_d_model(self):
        with pytest.raises(ValueError, match="n_heads"):
            _config(d_model=32, n_heads=5)

    def test_dropout_range(self):
        with pytest.raises(ValueError, match="dropout"):
            _config(dropout=1.0)

    def test_positive_fields(self):
        with pytest.raises(ValueError, match="vocab_size"):
            _config(vocab_size=0)

    def test_pool_mode_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="pool_mode"):
            _config(pool_mode="mean")

    def test_pooled_dim_depends_on_pool_mode(self):
        assert _config(d_model=32, pool_mode="dual").pooled_dim == 64
        assert _config(d_model=32, pool_mode="attn").pooled_dim == 32


class TestSealedEncoderModel:
    def test_forward_returns_dict_with_expected_shapes(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(3, cfg.max_seq_len, cfg.vocab_size)
        out = model(ids, mask)
        assert isinstance(out, dict)
        for name in ("score_play", "score_draw", "played_rate", "cast_lift"):
            assert out[name].shape == (3,)
        assert out["color_lift"].shape == (3, len(COLOR_ORDER))
        # forward() returns the pre-pool contextualized sequence, not the
        # full (B, T, vocab) logits — the caller applies mlm_head only at
        # the masked positions.
        assert out["contextualized"].shape == (3, cfg.max_seq_len, cfg.d_model)
        assert "mlm_logits" not in out

    def test_mlm_head_at_masked_positions_matches_full_then_index(self):
        # The MLM head is position-wise, so applying it to a gathered
        # subset of positions equals applying it to all positions and
        # then gathering — that equivalence is what lets _forward_batch
        # skip the wasted (B, T, vocab) projection.
        cfg = _config()
        torch.manual_seed(0)
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(2, cfg.max_seq_len, cfg.vocab_size)
        ctx = model(ids, mask)["contextualized"]
        positions = torch.zeros(2, cfg.max_seq_len, dtype=torch.bool)
        positions[0, 1] = positions[0, 3] = positions[1, 2] = True
        at_mask = model.mlm_head(ctx[positions])
        full = model.mlm_head(ctx)[positions]
        assert torch.allclose(at_mask, full, atol=1e-6)

    def test_played_rate_in_unit_interval(self):
        # FR-017 activation matching: played_rate uses sigmoid → [0, 1].
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(4, cfg.max_seq_len, cfg.vocab_size)
        out = model(ids, mask)
        assert torch.all((out["played_rate"] >= 0) & (out["played_rate"] <= 1))

    def test_signed_heads_in_minus_one_to_one(self):
        # Tanh range check.
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(4, cfg.max_seq_len, cfg.vocab_size)
        out = model(ids, mask)
        for name in ("score_play", "score_draw", "cast_lift"):
            assert torch.all((out[name] >= -1) & (out[name] <= 1))
        assert torch.all((out["color_lift"] >= -1) & (out["color_lift"] <= 1))

    def test_encode_shape_unchanged(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(2, cfg.max_seq_len, cfg.vocab_size)
        emb = model.encode(ids, mask)
        assert emb.shape == (2, 2 * cfg.d_model)

    def test_state_dict_contains_regression_heads_and_mlm_head(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        keys = set(model.state_dict().keys())
        assert any(k.startswith("regression_heads.") for k in keys)
        assert any(k.startswith("mlm_head.") for k in keys)

    def test_padding_mask_honored(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(2, cfg.max_seq_len, cfg.vocab_size)
        mask = mask.clone()
        mask[:, cfg.max_seq_len // 2:] = 0
        ids_padded = ids.clone()
        ids_padded[:, cfg.max_seq_len // 2:] = 0  # PAD id
        ids_garbage = ids_padded.clone()
        ids_garbage[:, cfg.max_seq_len // 2:] = (cfg.vocab_size - 1)
        out_padded = model.encode(ids_padded, mask)
        out_garbage = model.encode(ids_garbage, mask)
        assert torch.allclose(out_padded, out_garbage, atol=1e-5)

    def test_attn_pool_mode_halves_pooled_width(self):
        cfg = _config(pool_mode="attn")
        model = SealedEncoderModel(cfg)
        model.eval()
        # Regression heads project from d_model, not 2*d_model.
        assert model.regression_heads["score_play"][0].in_features == cfg.d_model
        ids, mask = _ids_and_mask(3, cfg.max_seq_len, cfg.vocab_size)
        assert model.encode(ids, mask).shape == (3, cfg.d_model)
        out = model(ids, mask)
        for name in ("score_play", "score_draw", "played_rate", "cast_lift"):
            assert out[name].shape == (3,)
        assert out["color_lift"].shape == (3, len(COLOR_ORDER))

    def test_random_init_unseeded(self):
        cfg = _config()
        torch.manual_seed(0)
        m1 = SealedEncoderModel(cfg)
        torch.manual_seed(1)
        m2 = SealedEncoderModel(cfg)
        diffs = [
            not torch.equal(p1, p2)
            for (p1, p2) in zip(m1.parameters(), m2.parameters())
        ]
        assert any(diffs)
