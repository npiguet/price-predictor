"""Tests for TransformerTrainingDataset."""

from __future__ import annotations

import math

import torch

from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset


def _make_tokenizer() -> MtgTokenizer:
    """Build a small tokenizer for tests."""
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "cardname": 2,
        "name": 3,
        "mana": 4,
        "cost": 5,
        "types": 6,
        "instant": 7,
        "creature": 8,
        "bear": 9,
        "legendary": 10,
        "planeswalker": 11,
        "jace": 12,
        "{R}": 13,
        "{1}": 14,
        "{G}": 15,
        "{2}": 16,
        "{U}": 17,
        "lightning": 18,
        "bolt": 19,
        "grizzly": 20,
        "bears": 21,
        "word": 22,
        "short": 23,
        "card": 24,
    }
    return MtgTokenizer(vocab)


_DEFAULT_PD = PrintingData.defaults()

SAMPLE_CARDS = [
    TrainingSample(
        "Lightning Bolt",
        "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
        "spell[1]: CARDNAME deals 3 damage to any target.",
        2.50,
        PrintingData(rarity="uncommon", printings_count=3, release_year=2018),
    ),
    TrainingSample(
        "Grizzly Bears",
        "name: grizzly bears\nmana cost: {1}{G}\ntypes: creature bear\n"
        "power toughness: 2/2",
        0.10,
        PrintingData(rarity="common", printings_count=5, release_year=2001),
    ),
    TrainingSample(
        "Jace, the Mind Sculptor",
        "name: jace, the mind sculptor\nmana cost: {2}{U}{U}\n"
        "types: legendary planeswalker jace\nloyalty: 3\n"
        "planeswalker[1]: [+2]: look at the top card of target "
        "player's library.",
        45.00,
        PrintingData(
            rarity="mythic", printings_count=2,
            release_year=2018, is_reserved=False,
        ),
    ),
]

SAMPLE_DEFAULT_PD_CARDS = [
    TrainingSample(s.name, s.text, s.price_eur, _DEFAULT_PD)
    for s in SAMPLE_CARDS
]


def _short_sample(name="Short", text="name: short card", price=1.0):
    return TrainingSample(name, text, price, _DEFAULT_PD)


class TestTransformerTrainingDataset:
    def test_length_matches_input(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target" in item
        assert "meta" in item

    def test_input_ids_shape(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        assert item["input_ids"].shape == (64,)

    def test_attention_mask_shape(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        assert item["attention_mask"].shape == (64,)

    def test_target_is_scalar(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        assert item["target"].shape == ()

    def test_meta_shape_with_defaults(self):
        """When sample has PrintingData.defaults(), meta is encoded from defaults."""
        from price_predictor.infrastructure.metadata_encoder import encode_metadata

        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            SAMPLE_DEFAULT_PD_CARDS, max_seq_len=64, tokenizer=tok,
        )
        item = ds[0]
        expected = encode_metadata(_DEFAULT_PD)
        assert item["meta"].shape == expected.shape
        assert torch.allclose(item["meta"], expected)

    def test_meta_shape_with_printing_data(self):
        """Meta should be encoded tensor of shape (15,)."""
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            SAMPLE_CARDS, max_seq_len=64, tokenizer=tok,
        )
        item = ds[0]
        assert item["meta"].shape == (15,)
        assert item["meta"].dtype == torch.float32

    def test_meta_values_differ_with_printing_data(self):
        """Meta tensors should reflect the PrintingData values."""
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            SAMPLE_CARDS, max_seq_len=64, tokenizer=tok,
        )
        meta_bolt = ds[0]["meta"]  # uncommon, printings=3
        meta_bears = ds[1]["meta"]  # common, printings=5
        # rarity slot differs: uncommon=0.33, common=0.0
        assert not torch.allclose(meta_bolt, meta_bears)

    def test_shifted_log_target_transform(self):
        """Target should be log(price + log_offset) using the configured offset."""
        tok = _make_tokenizer()
        log_offset = 0.5
        ds = TransformerTrainingDataset(
            SAMPLE_CARDS, max_seq_len=64, tokenizer=tok, log_offset=log_offset,
        )
        for i, sample in enumerate(SAMPLE_CARDS):
            expected = math.log(sample.price_eur + log_offset)
            actual = ds[i]["target"].item()
            assert abs(actual - expected) < 1e-5

    def test_input_ids_are_integers(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long

    def test_attention_mask_is_binary(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(SAMPLE_CARDS, max_seq_len=64, tokenizer=tok)
        item = ds[0]
        unique_vals = item["attention_mask"].unique()
        assert all(v in (0, 1) for v in unique_vals)

    def test_padding_produces_zeros_in_mask(self):
        """Short text padded to max_seq_len should have trailing zeros in mask."""
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            [_short_sample()], max_seq_len=64, tokenizer=tok,
        )
        mask = ds[0]["attention_mask"]
        assert (mask == 0).any()

    def test_truncation_to_max_seq_len(self):
        """Even with very long text, input_ids should not exceed max_seq_len."""
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            [_short_sample(name="Long Card", text="word " * 500)],
            max_seq_len=32, tokenizer=tok,
        )
        assert ds[0]["input_ids"].shape == (32,)

    def test_accepts_mtg_tokenizer_parameter(self):
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            SAMPLE_CARDS[:1], max_seq_len=32, tokenizer=tok,
        )
        assert len(ds) == 1

    def test_attention_mask_matches_padding_positions(self):
        """attention_mask should be 0 exactly where input_ids is PAD_ID (0)."""
        tok = _make_tokenizer()
        ds = TransformerTrainingDataset(
            [_short_sample()], max_seq_len=32, tokenizer=tok,
        )
        item = ds[0]
        ids = item["input_ids"]
        mask = item["attention_mask"]
        for i in range(len(ids)):
            if ids[i].item() == MtgTokenizer.PAD_ID:
                assert mask[i].item() == 0
