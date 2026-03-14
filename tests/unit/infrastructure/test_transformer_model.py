"""Tests for CardPriceTransformerModel (nn.Module)."""

from __future__ import annotations

import torch
import pytest

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel


def _make_config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=128, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, vocab_size=30522, dropout=0.1)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestCardPriceTransformerModelForward:
    """Test forward pass shapes and basic properties."""

    def test_output_shape_single_sample(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        output = model(input_ids, attention_mask)
        assert output.shape == (1,)

    def test_output_shape_batch(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        batch_size = 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        output = model(input_ids, attention_mask)
        assert output.shape == (batch_size,)

    def test_output_is_float(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        output = model(input_ids, attention_mask)
        assert output.dtype == torch.float32

    def test_deterministic_with_same_seed(self):
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
        attention_mask = torch.ones(2, config.max_seq_len)
        with torch.no_grad():
            out1 = model(input_ids, attention_mask)
            out2 = model(input_ids, attention_mask)
        assert torch.allclose(out1, out2)

    def test_attention_mask_affects_output(self):
        """Padding tokens (mask=0) should produce different output than real tokens (mask=1)."""
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        mask_full = torch.ones(1, config.max_seq_len)
        mask_half = torch.ones(1, config.max_seq_len)
        mask_half[0, config.max_seq_len // 2:] = 0
        with torch.no_grad():
            out_full = model(input_ids, mask_full)
            out_half = model(input_ids, mask_half)
        # Different masks should generally produce different outputs
        assert not torch.allclose(out_full, out_half)
