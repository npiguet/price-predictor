"""Tests for train_transformer use case."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAnalyzeSequenceLengths:
    def test_returns_multiple_of_8(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        texts = ["hello world"] * 100
        max_seq_len, stats = analyze_sequence_lengths(texts)
        assert max_seq_len % 8 == 0

    def test_minimum_is_64(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        texts = ["hi"] * 100  # Very short texts
        max_seq_len, stats = analyze_sequence_lengths(texts)
        assert max_seq_len >= 64

    def test_stats_contain_expected_keys(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        texts = ["hello world token test"] * 100
        _, stats = analyze_sequence_lengths(texts)
        assert "p95" in stats
        assert "p99" in stats
        assert "max" in stats
        assert "truncation_pct" in stats

    def test_truncation_pct_is_valid(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        texts = ["word " * 10] * 100
        _, stats = analyze_sequence_lengths(texts)
        assert 0.0 <= stats["truncation_pct"] <= 100.0

    def test_p95_less_than_or_equal_to_max(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        texts = ["word " * i for i in range(1, 101)]
        _, stats = analyze_sequence_lengths(texts)
        assert stats["p95"] <= stats["max"]


class TestTrainTransformer:
    @patch("price_predictor.application.train_transformer.parse_forge_cards")
    @patch("price_predictor.application.train_transformer.build_name_to_uuids")
    @patch("price_predictor.application.train_transformer.build_price_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    @patch("price_predictor.application.train_transformer.analyze_sequence_lengths")
    @patch("price_predictor.application.train_transformer.TransformerTrainingDataset")
    @patch("price_predictor.application.train_transformer.DataLoader")
    @patch("price_predictor.application.train_transformer.CardPriceTransformerModel")
    @patch("price_predictor.application.train_transformer.torch")
    @patch("price_predictor.application.train_transformer.train_test_split")
    @patch("price_predictor.application.train_transformer._train_loop")
    @patch("price_predictor.application.train_transformer.save_model")
    @patch("price_predictor.application.train_transformer.evaluate_transformer")
    def test_returns_train_result_with_expected_fields(
        self, mock_eval, mock_save, mock_train_loop, mock_split, mock_torch,
        mock_model_cls, mock_dataloader, mock_dataset_cls, mock_analyze,
        mock_match, mock_price_map, mock_name_uuids, mock_parse
    ):
        from price_predictor.application.train_transformer import train_transformer

        matched = [(f"Card {i}", f"name: card {i}", float(i + 1)) for i in range(20)]
        mock_parse.return_value = ([], [])
        mock_name_uuids.return_value = {}
        mock_price_map.return_value = {}
        mock_match.return_value = matched
        mock_analyze.return_value = (64, {"p95": 30, "p99": 40, "max": 50, "truncation_pct": 0.0})
        mock_split.return_value = (matched[:16], matched[16:])
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = MagicMock()
        mock_torch.manual_seed = MagicMock()
        mock_model_cls.return_value = MagicMock()
        mock_train_loop.return_value = (1, 0.1, False, 5)
        mock_save.return_value = Path("models/transformer/model.pt")
        mock_eval.return_value = MagicMock()

        result = train_transformer(
            output_dir=Path("output/"),
            forge_cards_path=Path("fake/forge"),
            prices_path=Path("fake/prices.json"),
            printings_path=Path("fake/printings.json"),
            model_output=Path("models/transformer/"),
        )

        assert result.card_count == 20
        assert result.max_seq_len == 64
        assert result.model_path == Path("models/transformer/model.pt")
        assert result.best_epoch == 1
        assert result.best_val_loss == 0.1

    @patch("price_predictor.application.train_transformer.parse_forge_cards")
    @patch("price_predictor.application.train_transformer.build_name_to_uuids")
    @patch("price_predictor.application.train_transformer.build_price_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    def test_insufficient_data_raises(
        self, mock_match, mock_price_map, mock_name_uuids, mock_parse
    ):
        from price_predictor.application.train_transformer import train_transformer

        mock_parse.return_value = ([], [])
        mock_name_uuids.return_value = {}
        mock_price_map.return_value = {}
        mock_match.return_value = [("Card 1", "text", 1.0)]  # Only 1 card

        with pytest.raises(ValueError, match="Insufficient"):
            train_transformer(
                output_dir=Path("output/"),
                forge_cards_path=Path("fake/forge"),
                prices_path=Path("fake/prices.json"),
                printings_path=Path("fake/printings.json"),
                model_output=Path("models/transformer/"),
            )
