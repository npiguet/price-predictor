"""Tests for the shared converted-card dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.application.converted_card_dataset import (
    ParsedCardDataset,
    load_parsed_cards,
    load_training_samples,
)
from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.value_objects import PrintingData


def _write_card(cards_dir: Path, filename: str, body: str) -> None:
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / filename).write_text(body, encoding="utf-8")


class TestLoadTrainingSamples:
    def test_returns_printing_data_from_metadata_map(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        _write_card(
            cards_dir, "test_card.txt",
            "name: Test Card\nmana cost: {1}{G}\ntypes: creature\npower toughness: 2/2\n",
        )

        expected_pd = PrintingData(
            is_reserved=True, rarity="rare", printings_count=5,
            release_year=2021, legalities=["commander", "modern", "legacy"],
        )
        samples = load_training_samples(
            cards_dir,
            price_map={"Test Card": 1.50},
            metadata_map={"Test Card": expected_pd},
        )

        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, TrainingSample)
        assert sample.name == "Test Card"
        assert sample.price_eur == 1.50
        assert sample.printing_data is expected_pd

    def test_no_metadata_map_uses_defaults(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        _write_card(
            cards_dir, "plain_card.txt",
            "name: Plain Card\nmana cost: {R}\ntypes: instant\n",
        )

        samples = load_training_samples(
            cards_dir, price_map={"Plain Card": 2.00}, metadata_map=None,
        )

        assert len(samples) == 1
        defaults = PrintingData.defaults()
        assert samples[0].printing_data.rarity == defaults.rarity
        assert samples[0].printing_data.printings_count == defaults.printings_count

    def test_text_is_raw_unmodified(self, tmp_path: Path) -> None:
        """The raw card text is returned untouched — no metadata lines injected."""
        cards_dir = tmp_path / "cards"
        body = "name: Test Card\nmana cost: {1}{G}\ntypes: creature\n"
        _write_card(cards_dir, "test_card.txt", body)

        samples = load_training_samples(
            cards_dir,
            price_map={"Test Card": 1.50},
            metadata_map={"Test Card": PrintingData(release_year=2021)},
        )
        text = samples[0].text
        assert "reserved:" not in text
        assert "rarity:" not in text
        assert "printings:" not in text
        assert "legalities:" not in text
        assert text == body

    def test_unknown_names_are_skipped(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        _write_card(cards_dir, "a.txt", "name: Known\ntypes: creature\n")
        _write_card(cards_dir, "b.txt", "name: Unknown\ntypes: creature\n")

        samples = load_training_samples(
            cards_dir, price_map={"Known": 1.0}, metadata_map=None,
        )
        assert [s.name for s in samples] == ["Known"]


class TestLoadParsedCards:
    def test_enriches_cards_with_printing_data(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        _write_card(
            cards_dir, "test_card.txt",
            "name: Test Card\nmana cost: {1}{G}\ntypes: creature\npower toughness: 2/2\n",
        )

        pd = PrintingData(rarity="mythic", printings_count=3, release_year=2020)
        dataset = load_parsed_cards(
            cards_dir,
            price_map={"Test Card": 1.50},
            metadata_map={"Test Card": pd},
        )

        assert isinstance(dataset, ParsedCardDataset)
        assert len(dataset.cards) == 1
        assert dataset.cards[0].printing_data is pd
        assert dataset.prices == [1.50]

    def test_records_skip_reasons(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        _write_card(cards_dir, "ok.txt", "name: Known\ntypes: creature\n")
        _write_card(cards_dir, "missing.txt", "name: Missing\ntypes: creature\n")

        dataset = load_parsed_cards(
            cards_dir, price_map={"Known": 1.0}, metadata_map=None,
        )
        assert dataset.skipped_reasons["no_printings_match"] == 1
        assert dataset.skipped_reasons["parse_error"] == 0
        assert dataset.total_skipped == 1

    def test_raises_when_directory_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            load_parsed_cards(tmp_path / "nope", price_map={}, metadata_map=None)
