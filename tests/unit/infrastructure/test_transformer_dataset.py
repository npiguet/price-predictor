"""Tests for TransformerTrainingDataset."""

from __future__ import annotations

import math

import torch
import pytest

from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset


SAMPLE_CARDS = [
    ("Lightning Bolt", "name: lightning bolt\nmana cost: {R}\ntypes: instant\nspell[1]: CARDNAME deals 3 damage to any target.", 2.50),
    ("Grizzly Bears", "name: grizzly bears\nmana cost: {1}{G}\ntypes: creature bear\npower toughness: 2/2", 0.10),
    ("Jace, the Mind Sculptor", "name: jace, the mind sculptor\nmana cost: {2}{U}{U}\ntypes: legendary planeswalker jace\nloyalty: 3\nplaneswalker[1]: [+2]: look at the top card of target player's library.", 45.00),
]


class TestTransformerTrainingDataset:
    def test_length_matches_input(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target" in item

    def test_input_ids_shape(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        assert item["input_ids"].shape == (64,)

    def test_attention_mask_shape(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        assert item["attention_mask"].shape == (64,)

    def test_target_is_scalar(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        assert item["target"].shape == ()

    def test_shifted_log_target_transform(self):
        """Target should be log(price + 2)."""
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        for i, (_, _, price) in enumerate(SAMPLE_CARDS):
            expected = math.log(price + 2)
            actual = ds[i]["target"].item()
            assert abs(actual - expected) < 1e-5, f"Card {i}: expected {expected}, got {actual}"

    def test_input_ids_are_integers(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long

    def test_attention_mask_is_binary(self):
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64)
        item = ds[0]
        unique_vals = item["attention_mask"].unique()
        assert all(v in (0, 1) for v in unique_vals)

    def test_padding_produces_zeros_in_mask(self):
        """Short text padded to max_seq_len should have trailing zeros in mask."""
        short_cards = [("Short", "name: short card", 1.0)]
        ds = TransformerTrainingDataset(short_cards, max_seq_len=64)
        mask = ds[0]["attention_mask"]
        # There should be some padding (zeros) for such a short text
        assert (mask == 0).any()

    def test_truncation_to_max_seq_len(self):
        """Even with very long text, input_ids should not exceed max_seq_len."""
        long_text = "word " * 500
        long_cards = [("Long Card", long_text, 1.0)]
        ds = TransformerTrainingDataset(long_cards, max_seq_len=32)
        assert ds[0]["input_ids"].shape == (32,)
