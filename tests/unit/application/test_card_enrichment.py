"""Tests for card text enrichment: append, extract, enrich-or-default."""

from __future__ import annotations

from price_predictor.application.card_enrichment import (
    enrich_card_text,
    enrich_or_default,
    extract_printing_data_from_text,
    extract_printing_fields_from_text,
)
from price_predictor.domain.value_objects import PrintingData

SAMPLE_CARD_TEXT = (
    "name: Grizzly Bears\n"
    "mana cost: {1}{G}\n"
    "types: creature bear\n"
    "power toughness: 2/2"
)


class TestEnrichCardText:
    def test_appends_five_lines(self) -> None:
        pd = PrintingData(
            is_reserved=True,
            rarity="common",
            printings_count=5,
            set_code="7ed",
            legalities=["commander", "legacy"],
        )
        result = enrich_card_text(SAMPLE_CARD_TEXT, pd)
        lines = result.splitlines()
        # Original text has 4 lines; enrichment adds 5 -> 9 total
        assert len(lines) == 9
        assert lines[4] == "reserved: true"
        assert lines[5] == "rarity: common"
        assert lines[6] == "printings: 5"
        assert lines[7] == "set: 7ed"
        assert lines[8] == "legalities: commander, legacy"

    def test_reserved_false(self) -> None:
        pd = PrintingData(
            is_reserved=False,
            rarity="rare",
            printings_count=1,
            set_code="ukn",
            legalities=[],
        )
        result = enrich_card_text(SAMPLE_CARD_TEXT, pd)
        assert "reserved: false" in result

    def test_empty_legalities(self) -> None:
        pd = PrintingData(legalities=[])
        result = enrich_card_text(SAMPLE_CARD_TEXT, pd)
        assert "legalities: " in result
        legalities_line = [
            line for line in result.splitlines()
            if line.startswith("legalities:")
        ][0]
        assert legalities_line == "legalities: "


class TestExtractPrintingFieldsFromText:
    def test_finds_existing_fields(self) -> None:
        text = SAMPLE_CARD_TEXT + "\nrarity: uncommon\nset: a25\n"
        fields = extract_printing_fields_from_text(text)
        assert fields["rarity"] == "uncommon"
        assert fields["set"] == "a25"

    def test_no_printing_fields_returns_empty(self) -> None:
        fields = extract_printing_fields_from_text(SAMPLE_CARD_TEXT)
        assert fields == {}

    def test_all_five_fields_detected(self) -> None:
        text = (
            SAMPLE_CARD_TEXT + "\n"
            "reserved: true\n"
            "rarity: mythic\n"
            "printings: 3\n"
            "set: wwk\n"
            "legalities: commander, legacy\n"
        )
        fields = extract_printing_fields_from_text(text)
        assert len(fields) == 5
        assert "reserved" in fields
        assert "rarity" in fields
        assert "printings" in fields
        assert "set" in fields
        assert "legalities" in fields


class TestExtractPrintingDataFromText:
    def test_returns_printing_data_when_fields_present(self) -> None:
        text = (
            SAMPLE_CARD_TEXT + "\n"
            "reserved: true\n"
            "rarity: mythic\n"
            "printings: 3\n"
            "set: wwk\n"
            "legalities: commander, legacy\n"
        )
        pd = extract_printing_data_from_text(text)
        assert pd is not None
        assert pd.is_reserved is True
        assert pd.rarity == "mythic"
        assert pd.printings_count == 3
        assert pd.set_code == "wwk"
        assert pd.legalities == ["commander", "legacy"]

    def test_returns_none_when_no_fields(self) -> None:
        pd = extract_printing_data_from_text(SAMPLE_CARD_TEXT)
        assert pd is None

    def test_partial_fields_use_defaults(self) -> None:
        text = SAMPLE_CARD_TEXT + "\nrarity: uncommon\n"
        pd = extract_printing_data_from_text(text)
        assert pd is not None
        assert pd.rarity == "uncommon"
        # Other fields get defaults
        assert pd.is_reserved is False
        assert pd.printings_count == 1
        assert pd.set_code == "ukn"
        assert pd.legalities == []

    def test_unrecognized_legalities_filtered_out(self) -> None:
        text = SAMPLE_CARD_TEXT + "\nlegalities: commander, kitchen_table, legacy\n"
        pd = extract_printing_data_from_text(text)
        assert pd is not None
        assert pd.legalities == ["commander", "legacy"]


class TestEnrichOrDefault:
    def test_known_card_auto_fills_all_fields(self) -> None:
        metadata_map = {
            "Grizzly Bears": PrintingData(
                is_reserved=False,
                rarity="common",
                printings_count=5,
                set_code="7ed",
                legalities=["commander", "legacy", "modern", "pauper", "penny", "vintage"],
            ),
        }
        result = enrich_or_default(SAMPLE_CARD_TEXT, metadata_map)
        assert "reserved: false" in result
        assert "rarity: common" in result
        assert "printings: 5" in result
        assert "set: 7ed" in result
        assert "legalities: commander, legacy, modern, pauper, penny, vintage" in result

    def test_unknown_card_applies_defaults(self) -> None:
        result = enrich_or_default(SAMPLE_CARD_TEXT, {})
        assert "reserved: false" in result
        assert "rarity: rare" in result
        assert "printings: 1" in result
        assert "set: ukn" in result
        # Defaults include all 10 recognized formats
        pd_defaults = PrintingData.defaults()
        expected_legalities = ", ".join(pd_defaults.legalities)
        assert f"legalities: {expected_legalities}" in result

    def test_client_override_respects_provided_rarity(self) -> None:
        text_with_rarity = SAMPLE_CARD_TEXT + "\nrarity: mythic\n"
        metadata_map = {
            "Grizzly Bears": PrintingData(
                is_reserved=False,
                rarity="common",
                printings_count=5,
                set_code="7ed",
                legalities=["commander", "legacy"],
            ),
        }
        result = enrich_or_default(text_with_rarity, metadata_map)
        # Client provided "mythic" should override the metadata "common"
        assert "rarity: mythic" in result
        # Other fields from metadata should still be used
        assert "set: 7ed" in result
        assert "printings: 5" in result

    def test_partial_fields_merge_client_and_autofill(self) -> None:
        text_with_partial = SAMPLE_CARD_TEXT + "\nreserved: true\nset: lea\n"
        metadata_map = {
            "Grizzly Bears": PrintingData(
                is_reserved=False,
                rarity="common",
                printings_count=5,
                set_code="7ed",
                legalities=["commander", "legacy"],
            ),
        }
        result = enrich_or_default(text_with_partial, metadata_map)
        # Client-provided fields override
        assert "reserved: true" in result
        assert "set: lea" in result
        # Auto-filled from metadata
        assert "rarity: common" in result
        assert "printings: 5" in result
        assert "legalities: commander, legacy" in result

    def test_all_five_fields_present_in_output(self) -> None:
        result = enrich_or_default(SAMPLE_CARD_TEXT, {})
        lines = result.splitlines()
        line_keys = set()
        for line in lines:
            colon_idx = line.find(":")
            if colon_idx > 0:
                key = line[:colon_idx].strip().lower()
                line_keys.add(key)
        for field_key in ("reserved", "rarity", "printings", "set", "legalities"):
            assert field_key in line_keys, f"Missing field: {field_key}"

    def test_case_insensitive_name_lookup(self) -> None:
        metadata_map = {
            "Grizzly Bears": PrintingData(
                is_reserved=False,
                rarity="common",
                printings_count=5,
                set_code="7ed",
                legalities=["commander"],
            ),
        }
        # Card text has lowercase name
        result = enrich_or_default(SAMPLE_CARD_TEXT, metadata_map)
        # Should still match despite case difference
        assert "rarity: common" in result
        assert "printings: 5" in result
