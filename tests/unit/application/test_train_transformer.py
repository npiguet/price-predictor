"""Tests for train_transformer use case."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_fixture_tokenizer(vocab_size: int = 512) -> "MtgTokenizer":
    """Build a fixture MtgTokenizer with a given vocab size."""
    from price_predictor.domain.tokenizer import MtgTokenizer

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i in range(vocab_size - 2):
        vocab[f"token_{i}"] = i + 2
    return MtgTokenizer(vocab)


class TestAnalyzeSequenceLengths:
    def test_returns_multiple_of_8(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        tokenizer = _make_fixture_tokenizer()
        texts = ["hello world"] * 100
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len % 8 == 0

    def test_minimum_is_64(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        tokenizer = _make_fixture_tokenizer()
        texts = ["hi"] * 100  # Very short texts
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len >= 64

    def test_stats_contain_expected_keys(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        tokenizer = _make_fixture_tokenizer()
        texts = ["hello world token test"] * 100
        _, stats = analyze_sequence_lengths(texts, tokenizer)
        assert "p95" in stats
        assert "p99" in stats
        assert "max" in stats

    def test_max_seq_len_covers_all_cards(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        tokenizer = _make_fixture_tokenizer()
        texts = ["word " * i for i in range(1, 101)]
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len >= stats["max"]

    def test_p95_less_than_or_equal_to_max(self):
        from price_predictor.application.train_transformer import analyze_sequence_lengths
        tokenizer = _make_fixture_tokenizer()
        texts = ["word " * i for i in range(1, 101)]
        _, stats = analyze_sequence_lengths(texts, tokenizer)
        assert stats["p95"] <= stats["max"]


class TestTrainTransformer:
    @patch("price_predictor.application.train_transformer.load_tokenizer")
    @patch("price_predictor.application.train_transformer.build_name_to_uuids")
    @patch("price_predictor.application.train_transformer.build_metadata_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    @patch("price_predictor.application.train_transformer.analyze_sequence_lengths")
    @patch("price_predictor.application.train_transformer.TransformerTrainingDataset")
    @patch("price_predictor.application.train_transformer.DataLoader")
    @patch("price_predictor.application.train_transformer.CardPriceTransformerModel")
    @patch("price_predictor.application.train_transformer.torch")
    @patch("price_predictor.application.train_transformer.train_test_split")
    @patch("price_predictor.application.train_transformer._train_loop")
    @patch("price_predictor.application.train_transformer.save_model")
    def test_returns_train_result_with_expected_fields(
        self, mock_save, mock_train_loop, mock_split, mock_torch,
        mock_model_cls, mock_dataloader, mock_dataset_cls, mock_analyze,
        mock_match, mock_metadata_map, mock_name_uuids, mock_load_tokenizer,
        tmp_path
    ):
        from price_predictor.application.train_transformer import train_transformer

        tokenizer = _make_fixture_tokenizer(vocab_size=512)
        mock_load_tokenizer.return_value = tokenizer

        matched = [(f"Card {i}", f"name: card {i}", float(i + 1)) for i in range(20)]
        mock_name_uuids.return_value = ({}, {})
        mock_metadata_map.return_value = ({}, {})
        mock_match.return_value = matched
        mock_analyze.return_value = (64, {"p95": 30, "p99": 40, "max": 50})
        mock_split.return_value = (matched[:16], matched[16:])
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = MagicMock()
        mock_torch.manual_seed = MagicMock()
        mock_model_cls.return_value = MagicMock()
        mock_train_loop.return_value = (1, 0.1, False, 5)
        mock_save.return_value = ("20260101-120000", Path("models/transformer/20260101-120000.pt"))

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[PAD]\n[UNK]\n", encoding="utf-8")

        result = train_transformer(
            output_dir=Path("output/"),
            prices_path=Path("fake/prices.json"),
            printings_path=Path("fake/printings.json"),
            model_output=Path("models/transformer/"),
            vocab_path=vocab_path,
        )

        assert result.card_count == 20
        assert result.max_seq_len == 64
        assert result.model_path == Path("models/transformer/20260101-120000.pt")
        assert result.best_epoch == 1
        assert result.best_val_loss == 0.1

    @patch("price_predictor.application.train_transformer.load_tokenizer")
    @patch("price_predictor.application.train_transformer.build_name_to_uuids")
    @patch("price_predictor.application.train_transformer.build_metadata_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    def test_vocab_size_equals_tokenizer_vocab_size(
        self, mock_match, mock_metadata_map, mock_name_uuids, mock_load_tokenizer,
        tmp_path
    ):
        """TransformerConfig.vocab_size should equal tokenizer.vocab_size (not 30522)."""
        from unittest.mock import call
        from price_predictor.application.train_transformer import train_transformer
        from price_predictor.domain.entities import TransformerConfig

        tokenizer = _make_fixture_tokenizer(vocab_size=512)
        mock_load_tokenizer.return_value = tokenizer

        matched = [(f"Card {i}", f"name: card {i}", float(i + 1)) for i in range(5)]
        mock_name_uuids.return_value = ({}, {})
        mock_metadata_map.return_value = ({}, {})
        mock_match.return_value = matched

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[PAD]\n[UNK]\n", encoding="utf-8")

        captured_configs = []

        with patch("price_predictor.application.train_transformer.analyze_sequence_lengths") as mock_analyze, \
             patch("price_predictor.application.train_transformer.TransformerTrainingDataset"), \
             patch("price_predictor.application.train_transformer.DataLoader"), \
             patch("price_predictor.application.train_transformer.CardPriceTransformerModel") as mock_model_cls, \
             patch("price_predictor.application.train_transformer.torch") as mock_torch, \
             patch("price_predictor.application.train_transformer.train_test_split") as mock_split, \
             patch("price_predictor.application.train_transformer._train_loop") as mock_loop, \
             patch("price_predictor.application.train_transformer.save_model") as mock_save:

            mock_analyze.return_value = (64, {"p95": 10, "p99": 15, "max": 20})
            mock_split.return_value = (matched[:4], matched[4:])
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = MagicMock()
            mock_torch.manual_seed = MagicMock()
            mock_loop.return_value = (1, 0.1, False, 5)
            mock_save.return_value = ("v1", Path("models/transformer/v1.pt"))

            def capture_config(config):
                captured_configs.append(config)
                return MagicMock()

            mock_model_cls.side_effect = capture_config

            train_transformer(
                output_dir=Path("output/"),
                prices_path=Path("fake/prices.json"),
                printings_path=Path("fake/printings.json"),
                model_output=Path("models/transformer/"),
                vocab_path=vocab_path,
            )

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.vocab_size == 512
        assert config.vocab_size != 30522

    @patch("price_predictor.application.train_transformer.build_name_to_uuids")
    @patch("price_predictor.application.train_transformer.build_metadata_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    def test_insufficient_data_raises(
        self, mock_match, mock_metadata_map, mock_name_uuids, tmp_path
    ):
        from price_predictor.application.train_transformer import train_transformer

        mock_name_uuids.return_value = ({}, {})
        mock_metadata_map.return_value = ({}, {})
        mock_match.return_value = [("Card 1", "text", 1.0)]  # Only 1 card

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[PAD]\n[UNK]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Insufficient"):
            with patch("price_predictor.application.train_transformer.load_tokenizer") as mock_tok:
                mock_tok.return_value = _make_fixture_tokenizer()
                train_transformer(
                    output_dir=Path("output/"),
                    prices_path=Path("fake/prices.json"),
                    printings_path=Path("fake/printings.json"),
                    model_output=Path("models/transformer/"),
                    vocab_path=vocab_path,
                )


class TestMatchCardsToTextsMetadata:
    """Tests for metadata enrichment in _match_cards_to_texts."""

    def test_metadata_map_enriches_text_with_printing_data_lines(self, tmp_path: Path):
        """When _match_cards_to_texts is called with a metadata_map, the returned
        texts contain printing data lines (reserved:, rarity:, printings:, set:, legalities:)."""
        from price_predictor.application.train_transformer import _match_cards_to_texts
        from price_predictor.domain.value_objects import PrintingData

        # Create a temporary text file with a card
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "test_card.txt").write_text(
            "name: Test Card\nmana cost: {1}{G}\ntypes: creature\npower toughness: 2/2\n",
            encoding="utf-8",
        )

        # Build a small name_to_uuids and price_map
        name_to_uuids = {"Test Card": ["uuid-1234"]}
        price_map = {"Test Card": 1.50}

        # Build a metadata_map
        metadata_map = {
            "Test Card": PrintingData(
                is_reserved=True,
                rarity="rare",
                printings_count=5,
                set_code="m21",
                legalities=["commander", "modern", "legacy"],
            ),
        }

        matched = _match_cards_to_texts(cards_dir, name_to_uuids, price_map, metadata_map)

        assert len(matched) == 1
        card_name, text, price = matched[0]
        assert card_name == "Test Card"
        assert price == 1.50

        # Verify the text contains printing data lines
        assert "reserved: true" in text
        assert "rarity: rare" in text
        assert "printings: 5" in text
        assert "set: m21" in text
        assert "legalities: commander, modern, legacy" in text

    def test_no_metadata_map_leaves_text_unchanged(self, tmp_path: Path):
        """When metadata_map is None, the text is returned without printing data lines."""
        from price_predictor.application.train_transformer import _match_cards_to_texts

        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        original_text = "name: Plain Card\nmana cost: {R}\ntypes: instant\n"
        (cards_dir / "plain_card.txt").write_text(original_text, encoding="utf-8")

        name_to_uuids = {"Plain Card": ["uuid-5678"]}
        price_map = {"Plain Card": 2.00}

        matched = _match_cards_to_texts(cards_dir, name_to_uuids, price_map, metadata_map=None)

        assert len(matched) == 1
        _, text, _ = matched[0]
        # No printing data lines should be present
        assert "reserved:" not in text
        assert "rarity:" not in text
        assert "printings:" not in text
        assert "set:" not in text
        assert "legalities:" not in text
