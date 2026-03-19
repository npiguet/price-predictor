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


def _make_meta(batch_size: int, meta_dim: int = 15) -> torch.Tensor:
    """Create a zero meta tensor for testing."""
    return torch.zeros(batch_size, meta_dim)


class TestCardPriceTransformerModelForward:
    """Test forward pass shapes and basic properties."""

    def test_output_shape_single_sample(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        meta = _make_meta(1)
        output = model(input_ids, attention_mask, meta)
        assert output.shape == (1,)

    def test_output_shape_batch(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        batch_size = 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size)
        output = model(input_ids, attention_mask, meta)
        assert output.shape == (batch_size,)

    def test_output_is_float(self):
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        meta = _make_meta(1)
        output = model(input_ids, attention_mask, meta)
        assert output.dtype == torch.float32

    def test_deterministic_with_same_seed(self):
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
        attention_mask = torch.ones(2, config.max_seq_len)
        meta = _make_meta(2)
        with torch.no_grad():
            out1 = model(input_ids, attention_mask, meta)
            out2 = model(input_ids, attention_mask, meta)
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
        meta = _make_meta(1)
        with torch.no_grad():
            out_full = model(input_ids, mask_full, meta)
            out_half = model(input_ids, mask_half, meta)
        # Different masks should generally produce different outputs
        assert not torch.allclose(out_full, out_half)

    def test_pooling_ignores_padding(self):
        """Both max and mean pooling must exclude masked (padding) positions."""
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        seq_len = config.max_seq_len

        # Build two inputs that differ only in their padding region
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        mask = torch.zeros(1, seq_len)
        mask[0, :seq_len // 2] = 1  # only first half is real

        # Second input has different token IDs in the masked region
        input_ids_alt = input_ids.clone()
        input_ids_alt[0, seq_len // 2:] = (input_ids_alt[0, seq_len // 2:] + 1) % config.vocab_size

        meta = _make_meta(1)
        with torch.no_grad():
            out1 = model(input_ids, mask, meta)
            out2 = model(input_ids_alt, mask, meta)

        # Outputs should be identical — masked positions must not contribute
        assert torch.allclose(out1, out2, atol=1e-5), (
            "Pooling did not ignore padding positions: outputs differ despite identical real tokens"
        )

    def test_output_head_is_sequential(self):
        """Regression head must be a Sequential with Linear → ReLU → Linear."""
        config = _make_config()
        model = CardPriceTransformerModel(config)
        head = model.output_head
        assert isinstance(head, torch.nn.Sequential), "output_head must be nn.Sequential"
        children = list(head.children())
        assert len(children) == 3, f"expected 3 layers in head, got {len(children)}"
        assert isinstance(children[0], torch.nn.Linear)
        assert isinstance(children[1], torch.nn.ReLU)
        assert isinstance(children[2], torch.nn.Linear)
        assert children[0].out_features == config.regression_hidden_dim
        assert children[2].out_features == 1

    def test_output_head_input_dim_includes_meta(self):
        """First Linear in output_head must accept 2*d_model + meta_dim inputs."""
        config = _make_config()
        model = CardPriceTransformerModel(config)
        head = model.output_head
        first_linear = list(head.children())[0]
        expected_in = 2 * config.d_model + config.meta_dim
        assert first_linear.in_features == expected_in

    def test_meta_affects_output(self):
        """Different meta vectors should produce different predictions."""
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        meta_zeros = torch.zeros(1, config.meta_dim)
        meta_ones = torch.ones(1, config.meta_dim)
        with torch.no_grad():
            out_zeros = model(input_ids, attention_mask, meta_zeros)
            out_ones = model(input_ids, attention_mask, meta_ones)
        assert not torch.allclose(out_zeros, out_ones)
