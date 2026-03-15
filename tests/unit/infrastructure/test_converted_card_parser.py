"""Tests for the converted card text parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.infrastructure.converted_card_parser import (
    parse_converted_cards,
    parse_converted_file,
    parse_converted_text,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "converted_cards"


# --- 1. Parse creature with power/toughness ---


def test_parse_creature_with_power_toughness():
    text = (FIXTURES / "grizzly_bears.txt").read_text()
    card = parse_converted_text(text)

    assert card.name == "grizzly bears"
    assert "Creature" in card.types
    assert "Bear" in card.subtypes
    assert card.mana_cost is not None
    assert card.mana_cost.total_mana_value == 2.0
    assert card.mana_cost.generic_mana == 1
    assert card.mana_cost.g == 1
    assert card.power == "2"
    assert card.toughness == "2"


# --- 2. Parse instant with spell ability ---


def test_parse_instant_with_spell_ability():
    text = (FIXTURES / "lightning_bolt.txt").read_text()
    card = parse_converted_text(text)

    assert card.name == "lightning bolt"
    assert "Instant" in card.types
    assert card.mana_cost is not None
    assert card.mana_cost.r == 1
    assert card.mana_cost.total_mana_value == 1.0
    assert card.ability_count == 1
    assert card.oracle_text is not None
    assert "3 damage" in card.oracle_text


# --- 3. Parse planeswalker with loyalty and multiple abilities ---


def test_parse_planeswalker_with_loyalty():
    text = (FIXTURES / "jace_the_mind_sculptor.txt").read_text()
    card = parse_converted_text(text)

    assert card.name == "jace, the mind sculptor"
    assert "Planeswalker" in card.types
    assert "Legendary" in card.supertypes
    assert "Jace" in card.subtypes
    assert card.mana_cost is not None
    assert card.mana_cost.u == 2
    assert card.mana_cost.generic_mana == 2
    assert card.mana_cost.total_mana_value == 4.0
    assert card.loyalty == "3"
    assert card.ability_count == 5
    assert card.oracle_text is not None


# --- 4. Missing optional fields still parse ---


def test_missing_optional_fields_still_parse():
    text = "name: Nameless One\ntypes: creature\n"
    card = parse_converted_text(text)

    assert card.name == "Nameless One"
    assert "Creature" in card.types
    assert card.mana_cost is None
    assert card.power is None
    assert card.toughness is None
    assert card.loyalty is None
    assert card.oracle_text is None
    assert card.ability_count == 0


# --- 5. Missing name raises ValueError ---


def test_missing_name_raises_value_error():
    text = "types: instant\nspell[1]: deal damage\n"
    with pytest.raises(ValueError, match="(?i)name"):
        parse_converted_text(text)


# --- 6. Missing types raises ValueError ---


def test_missing_types_raises_value_error():
    text = "name: Bad Card\nspell[1]: deal damage\n"
    with pytest.raises(ValueError, match="(?i)types"):
        parse_converted_text(text)


# --- 7. Empty text raises ValueError ---


def test_empty_text_raises_value_error():
    with pytest.raises(ValueError, match="(?i)empty"):
        parse_converted_text("")


def test_whitespace_only_raises_value_error():
    with pytest.raises(ValueError, match="(?i)empty"):
        parse_converted_text("   \n\n  ")


# --- 8. parse_converted_file with valid file ---


def test_parse_converted_file_valid():
    card = parse_converted_file(FIXTURES / "grizzly_bears.txt")

    assert card is not None
    assert card.name == "grizzly bears"
    assert "Creature" in card.types
    assert card.power == "2"
    assert card.toughness == "2"


def test_parse_converted_file_malformed_returns_none(tmp_path: Path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("This has no name or types fields")
    card = parse_converted_file(bad_file)

    assert card is None


def test_parse_converted_file_nonexistent_returns_none(tmp_path: Path):
    card = parse_converted_file(tmp_path / "does_not_exist.txt")

    assert card is None


# --- 9. parse_converted_cards with directory containing fixtures ---


def test_parse_converted_cards_with_fixtures():
    cards, errors = parse_converted_cards(FIXTURES)

    assert len(cards) == 3
    assert len(errors) == 0
    names = {c.name for c in cards}
    assert "lightning bolt" in names
    assert "grizzly bears" in names
    assert "jace, the mind sculptor" in names


# --- 10. parse_converted_cards with nonexistent directory ---


def test_parse_converted_cards_nonexistent_dir_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="(?i)does not exist"):
        parse_converted_cards(tmp_path / "nonexistent")


# --- 11. Empty output folder (no .txt files) raises ValueError ---


def test_parse_converted_cards_no_txt_files_raises(tmp_path: Path):
    # Directory exists but has no .txt files
    (tmp_path / "readme.md").write_text("not a card")
    with pytest.raises(ValueError, match="(?i)no .txt files"):
        parse_converted_cards(tmp_path)
