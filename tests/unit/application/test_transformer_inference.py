"""Tests for the shared transformer inference helpers."""

from __future__ import annotations

import math

import torch
from torch import nn

from price_predictor.application.transformer_inference import (
    predict_batch,
    predict_shifted_log,
    shifted_log_to_eur,
)
from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset


def _make_tokenizer() -> MtgTokenizer:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i, tok in enumerate(["name", "card", "test", "creature", "instant"], start=2):
        vocab[tok] = i
    return MtgTokenizer(vocab)


def _make_config() -> TransformerConfig:
    return TransformerConfig(
        d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=16, vocab_size=64, dropout=0.0,
    )


class _ConstantModel(nn.Module):
    """Model that always returns a fixed shifted-log prediction per batch row."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self._param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full((input_ids.size(0),), self.value) + self._param


class TestShiftedLogToEur:
    def test_inverts_shifted_log_transform(self):
        for price in [0.10, 1.0, 25.0, 100.0]:
            shifted = math.log(price + 2.0)
            assert abs(shifted_log_to_eur(shifted, log_offset=2.0) - price) < 1e-9

    def test_clamps_negative_to_zero(self):
        # exp(0) - 5 = -4 → clamped
        assert shifted_log_to_eur(0.0, log_offset=5.0) == 0.0


class TestPredictShiftedLog:
    def test_returns_model_output_value(self):
        model = _ConstantModel(value=1.5)
        config = _make_config()
        tok = _make_tokenizer()
        result = predict_shifted_log(model, tok, "name: test card", None, config)
        assert abs(result - 1.5) < 1e-6

    def test_uses_defaults_when_printing_data_none(self):
        """No printing_data should not raise — it's filled with defaults."""
        model = _ConstantModel(value=0.0)
        config = _make_config()
        tok = _make_tokenizer()
        predict_shifted_log(model, tok, "name: test card", None, config)

    def test_accepts_explicit_printing_data(self):
        model = _ConstantModel(value=2.0)
        config = _make_config()
        tok = _make_tokenizer()
        pd = PrintingData(rarity="mythic", printings_count=2, release_year=2020)
        result = predict_shifted_log(model, tok, "name: test card", pd, config)
        assert abs(result - 2.0) < 1e-6


class TestPredictBatch:
    def test_returns_predictions_and_targets_in_dataset_order(self):
        model = _ConstantModel(value=0.42)
        tok = _make_tokenizer()
        cards = [
            ("Card A", "name: card a", 1.0),
            ("Card B", "name: card b", 5.0),
            ("Card C", "name: card c", 25.0),
        ]
        dataset = TransformerTrainingDataset(
            cards, max_seq_len=16, tokenizer=tok, log_offset=2.0,
        )

        preds, targets = predict_batch(model, dataset, torch.device("cpu"))

        assert preds.shape == (3,)
        assert targets.shape == (3,)
        for value in preds:
            assert abs(float(value) - 0.42) < 1e-6
        for actual, (_, _, price) in zip(targets, cards):
            assert abs(float(actual) - math.log(price + 2.0)) < 1e-6
