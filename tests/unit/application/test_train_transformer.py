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


def _make_printing_data():
    from price_predictor.domain.value_objects import PrintingData
    return PrintingData(rarity="rare", printings_count=1, release_year=2020)


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
        from price_predictor.domain.value_objects import PrintingData

        tokenizer = _make_fixture_tokenizer(vocab_size=512)
        mock_load_tokenizer.return_value = tokenizer

        pd = _make_printing_data()
        matched = [(f"Card {i}", f"name: card {i}", float(i + 1), pd) for i in range(20)]
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
        from price_predictor.application.train_transformer import train_transformer
        from price_predictor.domain.entities import TransformerConfig

        tokenizer = _make_fixture_tokenizer(vocab_size=512)
        mock_load_tokenizer.return_value = tokenizer

        pd = _make_printing_data()
        matched = [(f"Card {i}", f"name: card {i}", float(i + 1), pd) for i in range(5)]
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
        from price_predictor.domain.value_objects import PrintingData

        mock_name_uuids.return_value = ({}, {})
        mock_metadata_map.return_value = ({}, {})
        pd = _make_printing_data()
        mock_match.return_value = [("Card 1", "text", 1.0, pd)]  # Only 1 card

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


class TestMatchCardsToTexts:
    """Tests for _match_cards_to_texts."""

    def test_returns_printing_data_from_metadata_map(self, tmp_path: Path):
        """When metadata_map is provided, the returned tuple includes the PrintingData."""
        from price_predictor.application.train_transformer import _match_cards_to_texts
        from price_predictor.domain.value_objects import PrintingData

        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "test_card.txt").write_text(
            "name: Test Card\nmana cost: {1}{G}\ntypes: creature\npower toughness: 2/2\n",
            encoding="utf-8",
        )

        name_to_uuids = {"Test Card": ["uuid-1234"]}
        price_map = {"Test Card": 1.50}
        expected_pd = PrintingData(
            is_reserved=True,
            rarity="rare",
            printings_count=5,
            release_year=2021,
            legalities=["commander", "modern", "legacy"],
        )
        metadata_map = {"Test Card": expected_pd}

        matched = _match_cards_to_texts(cards_dir, name_to_uuids, price_map, metadata_map)

        assert len(matched) == 1
        card_name, text, price, printing_data = matched[0]
        assert card_name == "Test Card"
        assert price == 1.50
        assert printing_data is expected_pd

    def test_no_metadata_map_returns_defaults(self, tmp_path: Path):
        """When metadata_map is None, PrintingData.defaults() is used."""
        from price_predictor.application.train_transformer import _match_cards_to_texts
        from price_predictor.domain.value_objects import PrintingData

        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        original_text = "name: Plain Card\nmana cost: {R}\ntypes: instant\n"
        (cards_dir / "plain_card.txt").write_text(original_text, encoding="utf-8")

        name_to_uuids = {"Plain Card": ["uuid-5678"]}
        price_map = {"Plain Card": 2.00}

        matched = _match_cards_to_texts(cards_dir, name_to_uuids, price_map, metadata_map=None)

        assert len(matched) == 1
        _, text, _, printing_data = matched[0]
        defaults = PrintingData.defaults()
        assert printing_data.rarity == defaults.rarity
        assert printing_data.printings_count == defaults.printings_count

    def test_text_does_not_contain_metadata_lines(self, tmp_path: Path):
        """Card text returned should NOT contain printing data lines — they're in the tensor."""
        from price_predictor.application.train_transformer import _match_cards_to_texts
        from price_predictor.domain.value_objects import PrintingData

        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "test_card.txt").write_text(
            "name: Test Card\nmana cost: {1}{G}\ntypes: creature\n",
            encoding="utf-8",
        )

        name_to_uuids = {"Test Card": ["uuid-1234"]}
        price_map = {"Test Card": 1.50}
        metadata_map = {
            "Test Card": PrintingData(rarity="rare", printings_count=5, release_year=2021),
        }

        matched = _match_cards_to_texts(cards_dir, name_to_uuids, price_map, metadata_map)
        _, text, _, _ = matched[0]

        # Text should be clean card text only
        assert "reserved:" not in text
        assert "rarity:" not in text
        assert "printings:" not in text
        assert "legalities:" not in text


# ─────────────────────────────────────────────────────────────────────────────
# T007 — Auxiliary label statistics computation and combined loss
# ─────────────────────────────────────────────────────────────────────────────

class TestAuxPosWeightComputation:
    """T007: pos_weight = num_neg / num_pos for each classification head."""

    def test_pos_weight_balanced(self):
        from price_predictor.application.train_transformer import _compute_pos_weight
        import numpy as np
        # 5 positives, 5 negatives → pos_weight = 1.0
        labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pw = _compute_pos_weight(labels)
        assert pw == pytest.approx(1.0)

    def test_pos_weight_imbalanced(self):
        from price_predictor.application.train_transformer import _compute_pos_weight
        import numpy as np
        # 1 positive, 9 negatives → pos_weight = 9.0
        labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pw = _compute_pos_weight(labels)
        assert pw == pytest.approx(9.0)

    def test_pos_weight_all_negative_returns_large(self):
        from price_predictor.application.train_transformer import _compute_pos_weight
        import numpy as np
        # All zeros → no positives, should not crash (return large number or 1.0)
        labels = np.zeros(10)
        pw = _compute_pos_weight(labels)
        assert pw >= 1.0


class TestRegressionStandardization:
    """T007: Regression targets are standardized with std floor of 1.0."""

    def test_std_floor_applied_when_std_below_one(self):
        from price_predictor.application.train_transformer import _compute_reg_stats
        import numpy as np
        # Near-constant values → std << 1, should use floor 1.0
        values = np.array([0.0, 0.0, 0.0, 0.0, 0.01])
        mean, std = _compute_reg_stats(values)
        assert std >= 1.0

    def test_std_above_one_not_floored(self):
        from price_predictor.application.train_transformer import _compute_reg_stats
        import numpy as np
        # Values with std > 1 should use actual std
        values = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        mean, std = _compute_reg_stats(values)
        assert std > 1.0

    def test_mean_computed_correctly(self):
        from price_predictor.application.train_transformer import _compute_reg_stats
        import numpy as np
        values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        mean, std = _compute_reg_stats(values)
        assert mean == pytest.approx(6.0)


class TestCombinedLoss:
    """T007: Combined loss = L_price + lambda * sum(L_aux_i)."""

    def test_combined_loss_formula(self):
        from price_predictor.application.train_transformer import _combined_loss
        import torch
        # L_price = 2.0, aux_losses = [1.0, 1.0], lambda = 0.5
        # Expected: 2.0 + 0.5 * 2.0 = 3.0
        l_price = torch.tensor(2.0)
        aux_losses = [torch.tensor(1.0), torch.tensor(1.0)]
        total = _combined_loss(l_price, aux_losses, aux_lambda=0.5)
        assert total.item() == pytest.approx(3.0)

    def test_combined_loss_zero_lambda_equals_price_loss(self):
        from price_predictor.application.train_transformer import _combined_loss
        import torch
        l_price = torch.tensor(2.5)
        aux_losses = [torch.tensor(1.0), torch.tensor(3.0)]
        total = _combined_loss(l_price, aux_losses, aux_lambda=0.0)
        assert total.item() == pytest.approx(2.5)


class TestSaveWrapperBase:
    """T007: After aux training, saving wrapper.base produces checkpoint loadable
    by load_model() with same state_dict keys as non-aux checkpoint (FR-007, SC-003)."""

    def test_save_base_produces_loadable_checkpoint(self, tmp_path):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        from price_predictor.infrastructure.transformer_store import save_model, load_model
        from price_predictor.domain.entities import TransformerConfig

        config = TransformerConfig(
            d_model=32, n_layers=1, n_heads=2, ff_dim=64,
            max_seq_len=16, vocab_size=100, dropout=0.0,
        )
        base = CardPriceTransformerModel(config)
        wrapper = AuxiliaryTrainingModel(base, n_aux=20)

        # Save base (not wrapper)
        _, model_path = save_model(wrapper.base, config, tmp_path)

        # Reload and compare state_dict keys
        loaded_model, loaded_config = load_model(tmp_path)
        assert set(loaded_model.state_dict().keys()) == set(base.state_dict().keys())

    def test_save_base_checkpoint_has_no_aux_head_keys(self, tmp_path):
        from price_predictor.infrastructure.transformer_model import (
            AuxiliaryTrainingModel,
            CardPriceTransformerModel,
        )
        from price_predictor.infrastructure.transformer_store import save_model, load_model
        from price_predictor.domain.entities import TransformerConfig

        config = TransformerConfig(
            d_model=32, n_layers=1, n_heads=2, ff_dim=64,
            max_seq_len=16, vocab_size=100, dropout=0.0,
        )
        base = CardPriceTransformerModel(config)
        wrapper = AuxiliaryTrainingModel(base, n_aux=20)

        _, model_path = save_model(wrapper.base, config, tmp_path)

        loaded_model, _ = load_model(tmp_path)
        keys = set(loaded_model.state_dict().keys())
        assert not any("aux_head" in k for k in keys)
