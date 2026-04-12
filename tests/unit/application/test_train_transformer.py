"""Tests for train_transformer use case."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from price_predictor.application.train_transformer import (
    _BestCheckpoint,
    _run_eval_epoch,
    _run_training_epoch,
    analyze_sequence_lengths,
    train_transformer,
)
from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData


def _make_fixture_tokenizer(vocab_size: int = 512) -> MtgTokenizer:
    """Build a fixture MtgTokenizer with a given vocab size."""
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i in range(vocab_size - 2):
        vocab[f"token_{i}"] = i + 2
    return MtgTokenizer(vocab)


def _make_printing_data() -> PrintingData:
    return PrintingData(rarity="rare", printings_count=1, release_year=2020)


class TestAnalyzeSequenceLengths:
    def test_returns_multiple_of_8(self):
        tokenizer = _make_fixture_tokenizer()
        texts = ["hello world"] * 100
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len % 8 == 0

    def test_minimum_is_64(self):
        tokenizer = _make_fixture_tokenizer()
        texts = ["hi"] * 100  # Very short texts
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len >= 64

    def test_stats_contain_expected_keys(self):
        tokenizer = _make_fixture_tokenizer()
        texts = ["hello world token test"] * 100
        _, stats = analyze_sequence_lengths(texts, tokenizer)
        assert "p95" in stats
        assert "p99" in stats
        assert "max" in stats

    def test_max_seq_len_covers_all_cards(self):
        tokenizer = _make_fixture_tokenizer()
        texts = ["word " * i for i in range(1, 101)]
        max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
        assert max_seq_len >= stats["max"]

    def test_p95_less_than_or_equal_to_max(self):
        tokenizer = _make_fixture_tokenizer()
        texts = ["word " * i for i in range(1, 101)]
        _, stats = analyze_sequence_lengths(texts, tokenizer)
        assert stats["p95"] <= stats["max"]


class _StubModel(nn.Module):
    """Tiny model with the (input_ids, attention_mask, meta) signature the epoch helpers expect."""

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
        best = _BestCheckpoint()
        improved = best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        assert improved is True
        assert best.best_epoch == 1
        assert best.best_val_loss == 0.5

    def test_higher_loss_does_not_improve(self):
        best = _BestCheckpoint()
        best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        improved = best.update(epoch=2, val_loss=0.7, model=nn.Linear(2, 1))
        assert improved is False
        assert best.best_epoch == 1

    def test_lower_loss_replaces_best(self):
        best = _BestCheckpoint()
        best.update(epoch=1, val_loss=0.5, model=nn.Linear(2, 1))
        improved = best.update(epoch=2, val_loss=0.3, model=nn.Linear(2, 1))
        assert improved is True
        assert best.best_epoch == 2
        assert best.best_val_loss == 0.3

    def test_restore_loads_snapshot_back_into_model(self):
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
        best = _BestCheckpoint()
        model = nn.Linear(2, 1)
        original = model.weight.detach().clone()
        best.restore(model)
        assert torch.allclose(model.weight, original)


class TestRunEpochHelpers:
    def test_train_mode_updates_weights(self):
        torch.manual_seed(0)
        model = _StubModel()
        before = model.linear.weight.detach().clone()
        loader = [_make_batch(target=10.0) for _ in range(3)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss = _run_training_epoch(model, loader, nn.MSELoss(), torch.device("cpu"), optimizer)

        assert isinstance(loss, float)
        assert not torch.allclose(model.linear.weight, before)

    def test_eval_mode_leaves_weights_unchanged(self):
        torch.manual_seed(0)
        model = _StubModel()
        before = model.linear.weight.detach().clone()
        loader = [_make_batch(target=10.0) for _ in range(3)]

        loss = _run_eval_epoch(model, loader, nn.MSELoss(), torch.device("cpu"))

        assert isinstance(loss, float)
        assert torch.allclose(model.linear.weight, before)

    def test_returns_mean_batch_loss(self):
        model = _StubModel()
        loader = [_make_batch(target=0.0) for _ in range(4)]

        loss = _run_eval_epoch(model, loader, nn.MSELoss(), torch.device("cpu"))

        assert loss >= 0.0


class TestTrainTransformer:
    @patch("price_predictor.application.train_transformer.build_metadata_map")
    @patch("price_predictor.application.train_transformer.load_training_samples")
    def test_insufficient_data_raises(
        self, mock_load_samples, mock_metadata_map, tmp_path
    ):
        mock_metadata_map.return_value = ({}, {})
        pd = _make_printing_data()
        mock_load_samples.return_value = [
            TrainingSample(name="Card 1", text="text", price_eur=1.0, printing_data=pd),
        ]

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
