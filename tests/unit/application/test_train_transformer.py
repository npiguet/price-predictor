"""Tests for train_transformer use case."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn


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


class _StubModel(nn.Module):
    """Tiny model with the (input_ids, attention_mask, meta) signature `_run_epoch` expects."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        x = input_ids.float().mean(dim=1, keepdim=True)
        return self.linear(x).squeeze(-1)


def _make_batch(target: float = 1.0) -> dict:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "target": torch.tensor([target], dtype=torch.float),
        "meta": torch.tensor([[0.0]], dtype=torch.float),
    }


class TestBestCheckpoint:
    def test_first_update_is_always_improvement(self):
        from price_predictor.application.train_transformer import _BestCheckpoint
        best = _BestCheckpoint()
        improved = best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        assert improved is True
        assert best.best_epoch == 1
        assert best.best_val_loss == 0.5

    def test_higher_loss_does_not_improve(self):
        from price_predictor.application.train_transformer import _BestCheckpoint
        best = _BestCheckpoint()
        best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        improved = best.update(epoch=2, val_loss=0.7, model=nn.Linear(2, 1))
        assert improved is False
        assert best.best_epoch == 1

    def test_lower_loss_replaces_best(self):
        from price_predictor.application.train_transformer import _BestCheckpoint
        best = _BestCheckpoint()
        best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        improved = best.update(epoch=2, val_loss=0.3, model=nn.Linear(2, 1))
        assert improved is True
        assert best.best_epoch == 2
        assert best.best_val_loss == 0.3

    def test_restore_loads_snapshot_back_into_model(self):
        from price_predictor.application.train_transformer import _BestCheckpoint
        torch.manual_seed(0)
        snapshot_model = nn.Linear(2, 1)
        snapshot_weight = snapshot_model.weight.detach().clone()

        best = _BestCheckpoint()
        best.update(epoch=1, val_loss=0.1, model=snapshot_model)

        with torch.no_grad():
            snapshot_model.weight.fill_(99.0)
        best.restore(snapshot_model)
        assert torch.allclose(snapshot_model.weight, snapshot_weight)

    def test_restore_is_noop_when_never_updated(self):
        from price_predictor.application.train_transformer import _BestCheckpoint
        best = _BestCheckpoint()
        model = nn.Linear(2, 1)
        original = model.weight.detach().clone()
        best.restore(model)
        assert torch.allclose(model.weight, original)


class TestRunEpoch:
    def test_train_mode_updates_weights(self):
        from price_predictor.application.train_transformer import _run_epoch
        torch.manual_seed(0)
        model = _StubModel()
        before = model.linear.weight.detach().clone()
        loader = [_make_batch(target=10.0) for _ in range(3)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss = _run_epoch(model, loader, nn.MSELoss(), torch.device("cpu"), optimizer=optimizer)

        assert isinstance(loss, float)
        assert not torch.allclose(model.linear.weight, before)

    def test_eval_mode_leaves_weights_unchanged(self):
        from price_predictor.application.train_transformer import _run_epoch
        torch.manual_seed(0)
        model = _StubModel()
        before = model.linear.weight.detach().clone()
        loader = [_make_batch(target=10.0) for _ in range(3)]

        loss = _run_epoch(model, loader, nn.MSELoss(), torch.device("cpu"))

        assert isinstance(loss, float)
        assert torch.allclose(model.linear.weight, before)

    def test_returns_mean_batch_loss(self):
        from price_predictor.application.train_transformer import _run_epoch
        model = _StubModel()
        loader = [_make_batch(target=0.0) for _ in range(4)]

        loss = _run_epoch(model, loader, nn.MSELoss(), torch.device("cpu"))

        assert loss >= 0.0


class TestTrainTransformer:
    @patch("price_predictor.application.train_transformer.build_metadata_map")
    @patch("price_predictor.application.train_transformer._match_cards_to_texts")
    def test_insufficient_data_raises(
        self, mock_match, mock_metadata_map, tmp_path
    ):
        from price_predictor.application.train_transformer import train_transformer

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

        price_map = {"Test Card": 1.50}
        expected_pd = PrintingData(
            is_reserved=True,
            rarity="rare",
            printings_count=5,
            release_year=2021,
            legalities=["commander", "modern", "legacy"],
        )
        metadata_map = {"Test Card": expected_pd}

        matched = _match_cards_to_texts(cards_dir, price_map, metadata_map)

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

        price_map = {"Plain Card": 2.00}

        matched = _match_cards_to_texts(cards_dir, price_map, metadata_map=None)

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

        price_map = {"Test Card": 1.50}
        metadata_map = {
            "Test Card": PrintingData(rarity="rare", printings_count=5, release_year=2021),
        }

        matched = _match_cards_to_texts(cards_dir, price_map, metadata_map)
        _, text, _, _ = matched[0]

        # Text should be clean card text only
        assert "reserved:" not in text
        assert "rarity:" not in text
        assert "printings:" not in text
        assert "legalities:" not in text
