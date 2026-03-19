"""Tests for PrintingData value object."""

from __future__ import annotations

import pytest

from price_predictor.domain.value_objects import (
    RECOGNIZED_FORMATS,
    PrintingData,
)


class TestPrintingDataConstruction:
    def test_valid_construction(self) -> None:
        pd = PrintingData(
            is_reserved=True,
            rarity="uncommon",
            printings_count=5,
            release_year=2018,
            legalities=["commander", "legacy", "modern"],
        )
        assert pd.is_reserved is True
        assert pd.rarity == "uncommon"
        assert pd.printings_count == 5
        assert pd.release_year == 2018
        assert pd.legalities == ["commander", "legacy", "modern"]

    def test_default_field_values(self) -> None:
        pd = PrintingData()
        assert pd.is_reserved is False
        assert pd.rarity == "rare"
        assert pd.printings_count == 1
        assert pd.release_year == 1993
        assert pd.legalities == []


class TestPrintingDataValidation:
    def test_invalid_rarity_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="rarity"):
            PrintingData(rarity="legendary")

    def test_printings_count_less_than_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="printings_count"):
            PrintingData(printings_count=0)

    def test_printings_count_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="printings_count"):
            PrintingData(printings_count=-3)

    def test_release_year_below_1993_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="release_year"):
            PrintingData(release_year=1992)

    def test_release_year_1993_is_valid(self) -> None:
        pd = PrintingData(release_year=1993)
        assert pd.release_year == 1993

    def test_unrecognized_format_in_legalities_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="legality.*not in RECOGNIZED_FORMATS"):
            PrintingData(legalities=["commander", "kitchen_table"])


class TestPrintingDataDefaults:
    def test_defaults_classmethod_returns_correct_values(self) -> None:
        d = PrintingData.defaults()
        assert d.is_reserved is False
        assert d.rarity == "rare"
        assert d.printings_count == 1
        assert d.release_year == 1993
        assert d.legalities == list(RECOGNIZED_FORMATS)

    def test_defaults_legalities_contains_all_recognized_formats(self) -> None:
        d = PrintingData.defaults()
        assert set(d.legalities) == set(RECOGNIZED_FORMATS)
        assert len(d.legalities) == len(RECOGNIZED_FORMATS)


class TestRecognizedFormats:
    def test_recognized_formats_has_exactly_10_entries(self) -> None:
        assert len(RECOGNIZED_FORMATS) == 10

    def test_recognized_formats_expected_entries(self) -> None:
        expected = {
            "standard", "pioneer", "modern", "brawl", "legacy",
            "vintage", "pauper", "commander", "penny", "oathbreaker",
        }
        assert set(RECOGNIZED_FORMATS) == expected
