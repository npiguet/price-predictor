"""Unit tests for the sealed encoder model."""

from __future__ import annotations

import pytest
import torch

from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel


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
    input_ids = torch.randint(low=2, high=vocab_size, size=(batch, seq_len))
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


class TestSealedEncoderModel:
    def test_forward_shape_is_batch(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(3, cfg.max_seq_len, cfg.vocab_size)
        out = model(ids, mask)
        assert out.shape == (3,)
        # sigmoid output bounded
        assert torch.all((out >= 0) & (out <= 1))

    def test_encode_shape_is_2x_d_model(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(2, cfg.max_seq_len, cfg.vocab_size)
        emb = model.encode(ids, mask)
        assert emb.shape == (2, 2 * cfg.d_model)

    def test_regression_head_present(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        keys = set(model.state_dict().keys())
        head_keys = {k for k in keys if k.startswith("regression_head.")}
        assert head_keys, "regression_head weights must exist on the live model"

    def test_padding_mask_honored(self):
        cfg = _config()
        model = SealedEncoderModel(cfg)
        model.eval()
        ids, mask = _ids_and_mask(2, cfg.max_seq_len, cfg.vocab_size)
        # Force second half of the sequence to padding
        mask = mask.clone()
        mask[:, cfg.max_seq_len // 2:] = 0
        ids_padded = ids.clone()
        ids_padded[:, cfg.max_seq_len // 2:] = 0  # PAD id
        # Replace the padded positions with garbage IDs; output should be unchanged.
        ids_garbage = ids_padded.clone()
        ids_garbage[:, cfg.max_seq_len // 2:] = (cfg.vocab_size - 1)
        out_padded = model.encode(ids_padded, mask)
        out_garbage = model.encode(ids_garbage, mask)
        assert torch.allclose(out_padded, out_garbage, atol=1e-5), (
            "encode output must not depend on tokens at masked-out positions"
        )

    def test_random_init_unseeded(self):
        """FR-016: encoder is randomly initialized — different seeds yield
        different weights."""
        cfg = _config()
        torch.manual_seed(0)
        m1 = SealedEncoderModel(cfg)
        torch.manual_seed(1)
        m2 = SealedEncoderModel(cfg)
        # at least one parameter must differ between the two models
        diffs = [
            not torch.equal(p1, p2)
            for (p1, p2) in zip(m1.parameters(), m2.parameters())
        ]
        assert any(diffs), "different seeds must produce different weights"
