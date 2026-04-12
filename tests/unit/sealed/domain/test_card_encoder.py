"""Unit tests for CardEncoder domain class."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sealed.domain.card_embedding_layout import DET_FEATURE_DIM, text_dim, total_dim
from sealed.domain.card_encoder import CardEncoder
from sealed.domain.deterministic_features import parse_deterministic_features


def _make_tokenizer(vocab_size: int = 50, max_seq_len: int = 16):
    """Build a mock MtgTokenizer that returns fixed-length sequences."""
    tok = MagicMock()
    tok.vocab_size = vocab_size

    def encode(text, max_length):
        ids = [1] * min(5, max_length) + [0] * max(0, max_length - 5)
        mask = [1] * min(5, max_length) + [0] * max(0, max_length - 5)
        return ids[:max_length], mask[:max_length]

    tok.encode.side_effect = encode
    return tok


def _make_model(d_model: int = 8, vocab_size: int = 50, max_seq_len: int = 16):
    """Build a real CardPriceTransformerModel in eval mode."""
    from price_predictor.domain.entities import TransformerConfig
    from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel

    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        dropout=0.0,
        meta_dim=4,
        regression_hidden_dim=8,
        log_offset=1.0,
    )
    model = CardPriceTransformerModel(config)
    model.eval()
    return model


class TestCardEncoderNameStripping:
    def test_name_line_stripped_before_tokenization(self):
        tok = _make_tokenizer()
        model = _make_model()
        encoder = CardEncoder(model, tok, max_seq_len=16)

        encoder.encode("name: Lightning Bolt\ninstant\ndeal 3 damage")
        call_text = tok.encode.call_args[0][0]
        assert not any(line.startswith("name:") for line in call_text.splitlines())

    def test_non_name_lines_preserved(self):
        tok = _make_tokenizer()
        model = _make_model()
        encoder = CardEncoder(model, tok, max_seq_len=16)

        encoder.encode("name: Foo\ntype: Creature\npt: 2/2")
        call_text = tok.encode.call_args[0][0]
        assert "type: Creature" in call_text
        assert "pt: 2/2" in call_text

    def test_card_without_name_line_passes_through(self):
        tok = _make_tokenizer()
        model = _make_model()
        encoder = CardEncoder(model, tok, max_seq_len=16)

        encoder.encode("type: Land\nbasicLandType: Forest")
        call_text = tok.encode.call_args[0][0]
        assert "type: Land" in call_text


class TestCardEncoderOutputShape:
    def test_returns_numpy_array(self):
        tok = _make_tokenizer()
        model = _make_model(d_model=8)
        encoder = CardEncoder(model, tok, max_seq_len=16)
        result = encoder.encode("name: Test\ntype: Creature")
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_layout_total_dim(self):
        d_model = 8
        tok = _make_tokenizer()
        model = _make_model(d_model=d_model)
        encoder = CardEncoder(model, tok, max_seq_len=16)
        result = encoder.encode("name: Test\ntype: Creature")
        assert result.shape == (total_dim(d_model),)

    def test_output_dtype_is_float32(self):
        tok = _make_tokenizer()
        model = _make_model(d_model=8)
        encoder = CardEncoder(model, tok, max_seq_len=16)
        result = encoder.encode("name: Test\ntype: Creature")
        assert result.dtype == np.float32


class TestCardEncoderConcatenatedLayout:
    """CardEncoder concatenates the text embedding with deterministic features."""

    def test_output_matches_total_dim(self):
        d_model = 256
        tok = _make_tokenizer()
        model = _make_model(d_model=d_model)
        encoder = CardEncoder(model, tok, max_seq_len=16)
        result = encoder.encode("name: Test\nmana cost: {R}\ntypes: creature\npower toughness: 2/2")
        assert result.shape == (total_dim(d_model),)

    def test_text_slice_matches_text_embedding(self):
        d_model = 256
        tok = _make_tokenizer()
        model = _make_model(d_model=d_model)
        encoder = CardEncoder(model, tok, max_seq_len=16)

        card_text = "name: Test\nmana cost: {R}\ntypes: creature\npower toughness: 2/2"
        full_result = encoder.encode(card_text)
        text_embedding = full_result[:text_dim(d_model)]

        # Re-encode using model directly to get just the text embedding
        lines = [line for line in card_text.splitlines() if not line.startswith("name:")]
        text = "\n".join(lines)
        input_ids, attention_mask = tok.encode(text, 16)
        import torch
        ids_t = torch.tensor([input_ids], dtype=torch.long)
        mask_t = torch.tensor([attention_mask], dtype=torch.long)
        expected = model.encode(ids_t, mask_t).squeeze(0).cpu().numpy().astype(np.float32)

        np.testing.assert_array_equal(text_embedding, expected)

    def test_deterministic_slice_matches_parsed_features(self):
        d_model = 256
        tok = _make_tokenizer()
        model = _make_model(d_model=d_model)
        encoder = CardEncoder(model, tok, max_seq_len=16)

        card_text = "name: Test\nmana cost: {R}\ntypes: creature\npower toughness: 2/2"
        full_result = encoder.encode(card_text)
        det_feats = full_result[text_dim(d_model):]
        assert det_feats.shape == (DET_FEATURE_DIM,)

        expected = parse_deterministic_features(card_text)
        np.testing.assert_array_equal(det_feats, expected)


class TestCardEncoderTokenization:
    def test_max_seq_len_passed_to_tokenizer(self):
        seq_len = 16
        tok = _make_tokenizer(max_seq_len=seq_len)
        model = _make_model(max_seq_len=seq_len)
        encoder = CardEncoder(model, tok, max_seq_len=seq_len)

        encoder.encode("some text")
        tok.encode.assert_called_once()
        assert tok.encode.call_args[0][1] == seq_len
