"""Tests for CardPriceTransformerModel and AuxiliaryTrainingModel (nn.Module)."""

from __future__ import annotations

import torch
import pytest

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel

# Minimal ordinal class maps for tests: 2 classes per ordinal head (indices 7-13)
_TEST_ORDINAL_CLASS_MAPS: dict[int, list[float]] = {i: [0.0, 1.0] for i in range(7, 14)}


def _make_config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=128, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, vocab_size=30522, dropout=0.1)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_meta(batch_size: int, meta_dim: int = 30) -> torch.Tensor:
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


# ─────────────────────────────────────────────────────────────────────────────
# T005 — _embed() and AuxiliaryTrainingModel
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbed:
    """T005: _embed() extracts the shared pooled representation."""

    def test_embed_output_shape(self):
        from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
        config = _make_config()
        model = CardPriceTransformerModel(config)
        model.eval()
        batch_size = 3
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        with torch.no_grad():
            pooled = model._embed(input_ids, attention_mask)
        assert pooled.shape == (batch_size, 2 * config.d_model)

    def test_embed_matches_encode_output(self):
        """_embed() (no_grad) must produce same values as encode()."""
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
        attention_mask = torch.ones(2, config.max_seq_len)
        encode_out = model.encode(input_ids, attention_mask)
        with torch.no_grad():
            embed_out = model._embed(input_ids, attention_mask)
        assert torch.allclose(encode_out, embed_out, atol=1e-6)

    def test_embed_has_gradients(self):
        """_embed() should allow gradients to flow (unlike encode which has no_grad)."""
        config = _make_config(dropout=0.0)
        model = CardPriceTransformerModel(config)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        attention_mask = torch.ones(1, config.max_seq_len)
        pooled = model._embed(input_ids, attention_mask)
        # Sum and backward must not raise
        pooled.sum().backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestAuxiliaryTrainingModel:
    """T005: AuxiliaryTrainingModel wraps base model with 20 auxiliary heads."""

    def test_forward_returns_tuple(self):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        aux_model.eval()
        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size, config.meta_dim)
        with torch.no_grad():
            price_pred, aux_preds = aux_model(input_ids, attention_mask, meta)
        assert isinstance(price_pred, torch.Tensor)
        assert isinstance(aux_preds, list)

    def test_price_pred_shape(self):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        aux_model.eval()
        batch_size = 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size, config.meta_dim)
        with torch.no_grad():
            price_pred, _ = aux_model(input_ids, attention_mask, meta)
        assert price_pred.shape == (batch_size,)

    def test_aux_preds_count_equals_n_aux(self):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        n_aux = 20
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        aux_model.eval()
        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size, config.meta_dim)
        with torch.no_grad():
            _, aux_preds = aux_model(input_ids, attention_mask, meta)
        assert len(aux_preds) == n_aux

    def test_each_aux_pred_shape(self):
        """Classification heads → (batch,); ordinal heads → (batch, K)."""
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        aux_model.eval()
        batch_size = 3
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size, config.meta_dim)
        with torch.no_grad():
            _, aux_preds = aux_model(input_ids, attention_mask, meta)
        for i, pred in enumerate(aux_preds):
            if i in _TEST_ORDINAL_CLASS_MAPS:
                K = len(_TEST_ORDINAL_CLASS_MAPS[i])
                assert pred.shape == (batch_size, K), f"aux_pred[{i}] ordinal shape mismatch"
            else:
                assert pred.shape == (batch_size,), f"aux_pred[{i}] classification shape mismatch"

    def test_gradients_flow_through_aux_heads(self):
        """Gradients from aux heads must reach encoder parameters."""
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config(dropout=0.0)
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        aux_model.train()
        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        attention_mask = torch.ones(batch_size, config.max_seq_len)
        meta = _make_meta(batch_size, config.meta_dim)
        _, aux_preds = aux_model(input_ids, attention_mask, meta)
        # Sum all aux predictions and backprop
        sum(p.sum() for p in aux_preds).backward()
        # Encoder parameters should have gradients
        encoder_has_grad = any(
            p.grad is not None
            for p in base.encoder.parameters()
        )
        assert encoder_has_grad, "Gradients must flow through aux heads to encoder"

    def test_aux_heads_count(self):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        assert len(aux_model.aux_heads) == 20

    def test_base_attribute_is_original_model(self):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        config = _make_config()
        base = CardPriceTransformerModel(config)
        aux_model = AuxiliaryTrainingModel(base, _TEST_ORDINAL_CLASS_MAPS)
        assert aux_model.base is base


# ─────────────────────────────────────────────────────────────────────────────
# encode_mana_features
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeManaFeatures:
    """Tests for encode_mana_features() in metadata_encoder."""

    def test_returns_tensor_of_shape_15(self):
        from price_predictor.infrastructure.metadata_encoder import encode_mana_features
        result = encode_mana_features("name: lightning bolt\nmana cost: {R}\ntypes: instant\n")
        assert result.shape == (15,)

    def test_dtype_is_float32(self):
        from price_predictor.infrastructure.metadata_encoder import encode_mana_features
        result = encode_mana_features("name: foo\n")
        assert result.dtype == torch.float32

    def test_values_in_0_1_range(self):
        from price_predictor.infrastructure.metadata_encoder import encode_mana_features
        result = encode_mana_features("name: baneslayer\nmana cost: {3}{W}{W}\ntypes: creature\n")
        assert (result >= 0.0).all() and (result <= 1.0).all()

    def test_red_spell_has_nonzero_pip_r(self):
        from price_predictor.infrastructure.metadata_encoder import encode_mana_features
        result = encode_mana_features("name: lightning bolt\nmana cost: {R}\ntypes: instant\n")
        # pip_R is index 3 of the mana features
        assert result[3].item() > 0.0

    def test_combined_meta_is_30_dims(self):
        from price_predictor.infrastructure.metadata_encoder import encode_metadata, encode_mana_features
        from price_predictor.domain.value_objects import PrintingData
        import torch
        pd = PrintingData.defaults()
        full_meta = torch.cat([encode_metadata(pd), encode_mana_features("name: test\n")])
        assert full_meta.shape == (30,)
