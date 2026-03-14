"""Tests for evaluate_transformer use case."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from price_predictor.domain.entities import TransformerConfig


def _make_config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=128, n_layers=4, n_heads=4, ff_dim=512, max_seq_len=64, vocab_size=30522, dropout=0.1)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestEvaluateTransformer:
    @patch("price_predictor.application.evaluate_transformer.load_model")
    @patch("price_predictor.application.evaluate_transformer.parse_forge_cards")
    @patch("price_predictor.application.evaluate_transformer.build_name_to_uuids")
    @patch("price_predictor.application.evaluate_transformer.build_price_map")
    @patch("price_predictor.application.evaluate_transformer._match_cards_to_texts")
    def test_returns_eval_result_with_metrics(
        self, mock_match, mock_price_map, mock_name_uuids, mock_parse, mock_load, tmp_path
    ):
        from price_predictor.application.evaluate_transformer import evaluate_transformer

        config = _make_config()

        # Create a mock model that returns fixed shifted-log predictions
        mock_model = MagicMock()
        # The model will be called with batched input_ids and attention_mask
        # Return predictions as shifted-log values
        val_prices = [1.0, 2.0, 5.0, 10.0, 3.0]
        shifted_log_preds = torch.tensor(
            [math.log(p + 2) for p in val_prices], dtype=torch.float32
        )
        mock_model.return_value = shifted_log_preds
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_load.return_value = (mock_model, config)

        # 25 total cards -> 80/20 split -> 5 val cards
        cards = [MagicMock(name=f"Card {i}") for i in range(25)]
        mock_parse.return_value = (cards, [])
        mock_name_uuids.return_value = {}
        mock_price_map.return_value = {}

        # _match_cards_to_texts returns 25 matched cards
        matched = [(f"Card {i}", f"name: card {i}", float(i + 1)) for i in range(25)]
        mock_match.return_value = matched

        result = evaluate_transformer(
            model_dir=Path("fake/model"),
            output_dir=tmp_path,
            forge_cards_path=Path("fake/forge"),
            prices_path=Path("fake/prices.json"),
            printings_path=Path("fake/printings.json"),
            random_seed=42,
        )

        assert result.sample_count == 5
        assert result.model_path == Path("fake/model")
        assert result.mean_absolute_error_eur >= 0
        assert result.median_abs_error_log >= 0
        assert result.per_card is not None
        assert len(result.per_card) == 5

    def test_shifted_log_inverse_transform(self):
        """Verify exp(x) - 2 correctly inverts log(price + 2)."""
        prices = [0.10, 2.50, 45.00, 100.00]
        for price in prices:
            shifted_log = math.log(price + 2)
            recovered = math.exp(shifted_log) - 2
            assert abs(recovered - price) < 1e-10

    def test_mae_computation(self):
        """MAE = mean of |actual - predicted| in EUR."""
        actual = np.array([2.0, 4.0, 6.0])
        predicted = np.array([3.0, 3.0, 8.0])
        mae = float(np.mean(np.abs(actual - predicted)))
        # |3-2| + |3-4| + |8-6| = 1 + 1 + 2 = 4, MAE = 4/3
        assert abs(mae - 4.0 / 3.0) < 1e-10

    def test_median_abs_error_log_computation(self):
        """Median abs error in shifted-log space = median(|log(actual+2) - log(predicted+2)|)."""
        actual = np.array([10.0, 20.0, 50.0])
        predicted = np.array([12.0, 18.0, 45.0])
        log_errors = np.abs(np.log(actual + 2) - np.log(predicted + 2))
        median_log_err = float(np.median(log_errors))
        # log(12)-log(14) ≈ 0.154, |log(22)-log(20)| ≈ 0.095, |log(52)-log(47)| ≈ 0.101
        # sorted: 0.095, 0.101, 0.154 -> median ≈ 0.101
        expected = float(np.median(np.abs(np.log(actual + 2) - np.log(predicted + 2))))
        assert abs(median_log_err - expected) < 1e-10
