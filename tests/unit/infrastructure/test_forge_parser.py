"""Tests for Forge card script parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.domain.entities import Card
from price_predictor.infrastructure.forge_parser import (
    parse_forge_cards,
    parse_forge_file,
    parse_forge_text,
)


@pytest.fixture
def forge_cards_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "forge_cards"


class TestParseForgeFile:
    def test_parse_vanilla_creature(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "grizzly_bears.txt")
        assert card is not None
        assert card.name == "Grizzly Bears"
        assert "Creature" in card.types
        assert "Bear" in card.subtypes
        assert card.mana_cost is not None
        assert card.mana_cost.total_mana_value == 2.0
        assert card.mana_cost.g == 1
        assert card.mana_cost.generic_mana == 1
        assert card.power == "2"
        assert card.toughness == "2"
        assert card.layout == "normal"

    def test_parse_legendary_creature_with_keywords(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "ragavan_nimble_pilferer.txt")
        assert card is not None
        assert card.name == "Ragavan, Nimble Pilferer"
        assert "Legendary" in card.supertypes
        assert "Creature" in card.types
        assert "Monkey" in card.subtypes
        assert "Pirate" in card.subtypes
        assert "Dash" in card.keywords
        assert card.power == "2"
        assert card.toughness == "1"

    def test_parse_planeswalker_with_loyalty(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "jace_the_mind_sculptor.txt")
        assert card is not None
        assert card.name == "Jace, the Mind Sculptor"
        assert "Planeswalker" in card.types
        assert "Legendary" in card.supertypes
        assert card.loyalty == "3"
        assert card.ability_count == 4

    def test_parse_instant(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "lightning_bolt.txt")
        assert card is not None
        assert card.name == "Lightning Bolt"
        assert "Instant" in card.types
        assert card.mana_cost is not None
        assert card.mana_cost.r == 1
        assert card.ability_count == 1

    def test_parse_land_no_cost(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "island.txt")
        assert card is not None
        assert card.name == "Island"
        assert "Land" in card.types
        assert "Basic" in card.supertypes
        assert card.mana_cost is None

    def test_parse_colorless_mana_card(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "matter_reshaper.txt")
        assert card is not None
        assert card.name == "Matter Reshaper"
        assert card.mana_cost is not None
        assert card.mana_cost.colorless_mana == 1
        assert card.mana_cost.generic_mana == 2
        assert card.mana_cost.total_mana_value == 3.0
        assert card.mana_cost.color_count == 0

    def test_parse_transform_card_front_face_only(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(
            forge_cards_dir / "delver_of_secrets_insectile_aberration.txt"
        )
        assert card is not None
        assert card.name == "Delver of Secrets"
        assert card.power == "1"
        assert card.toughness == "1"
        assert card.layout == "doublefaced"
        # Should NOT have Flying from back face
        assert "Flying" not in card.keywords

    def test_parse_split_card_front_face_only(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "fire_ice.txt")
        assert card is not None
        assert card.name == "Fire"
        assert "Instant" in card.types
        assert card.layout == "split"
        assert card.mana_cost is not None
        assert card.mana_cost.r == 1

    def test_parse_creature_with_keywords_stripped(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "serra_angel.txt")
        assert card is not None
        assert "Flying" in card.keywords
        assert "Vigilance" in card.keywords
        assert card.power == "4"
        assert card.toughness == "4"

    def test_parse_artifact(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "sol_ring.txt")
        assert card is not None
        assert "Artifact" in card.types
        assert card.mana_cost is not None
        assert card.mana_cost.generic_mana == 1

    def test_parse_scheme(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "your_puny_minds_cannot_fathom.txt")
        assert card is not None
        assert card.name == "Your Puny Minds Cannot Fathom"
        assert "Scheme" in card.types
        assert card.mana_cost is None
        assert card.oracle_text is not None

    def test_parse_plane(self, forge_cards_dir: Path) -> None:
        card = parse_forge_file(forge_cards_dir / "academy_at_tolaria_west.txt")
        assert card is not None
        assert card.name == "Academy at Tolaria West"
        assert "Plane" in card.types
        assert "Dominaria" in card.subtypes
        assert card.mana_cost is None
        assert card.oracle_text is not None

    def test_malformed_file_returns_none(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_card.txt"
        bad_file.write_text("This is not a valid card file\nNo Name field here")
        card = parse_forge_file(bad_file)
        assert card is None


class TestParseForgeText:
    """Tests for parse_forge_text() — parses Forge card script strings."""

    def test_parse_complete_card_script(self, forge_cards_dir: Path) -> None:
        text = (forge_cards_dir / "ragavan_nimble_pilferer.txt").read_text()
        card = parse_forge_text(text)
        assert card.name == "Ragavan, Nimble Pilferer"
        assert "Legendary" in card.supertypes
        assert "Creature" in card.types
        assert "Monkey" in card.subtypes
        assert "Pirate" in card.subtypes
        assert "Dash" in card.keywords
        assert card.mana_cost is not None
        assert card.mana_cost.r == 1
        assert card.power == "2"
        assert card.toughness == "1"

    def test_parse_instant_script(self, forge_cards_dir: Path) -> None:
        text = (forge_cards_dir / "lightning_bolt.txt").read_text()
        card = parse_forge_text(text)
        assert card.name == "Lightning Bolt"
        assert "Instant" in card.types
        assert card.mana_cost is not None
        assert card.mana_cost.r == 1

    def test_parse_partial_script_only_types(self) -> None:
        text = "Name:Minimal Card\nTypes:Instant\n"
        card = parse_forge_text(text)
        assert card.name == "Minimal Card"
        assert "Instant" in card.types
        assert card.mana_cost is None

    def test_empty_string_raises_error(self) -> None:
        with pytest.raises(ValueError, match="(?i)empty"):
            parse_forge_text("")

    def test_malformed_string_no_types_raises_error(self) -> None:
        with pytest.raises(ValueError, match="(?i)types"):
            parse_forge_text("Name:Bad Card\nManaCost:R\n")

    def test_matches_parse_forge_file(self, forge_cards_dir: Path) -> None:
        """parse_forge_text(file.read_text()) produces same result as parse_forge_file(file)."""
        path = forge_cards_dir / "grizzly_bears.txt"
        from_file = parse_forge_file(path)
        from_text = parse_forge_text(path.read_text())
        assert from_file is not None
        assert from_text.name == from_file.name
        assert from_text.types == from_file.types
        assert from_text.power == from_file.power
        assert from_text.toughness == from_file.toughness


class TestParseForgeCards:
    def test_parses_all_valid_cards(self, forge_cards_dir: Path) -> None:
        cards, errors = parse_forge_cards(forge_cards_dir)
        assert len(cards) >= 10
        assert isinstance(cards[0], Card)

    def test_reports_errors_for_bad_files(self, tmp_path: Path) -> None:
        # Create a mix of good and bad files
        good = tmp_path / "good.txt"
        good.write_text("Name:Test Card\nManaCost:R\nTypes:Instant\n")
        bad = tmp_path / "bad.txt"
        bad.write_text("not a card\n")
        cards, errors = parse_forge_cards(tmp_path)
        assert len(cards) == 1
        assert len(errors) >= 1

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        cards, errors = parse_forge_cards(tmp_path / "nonexistent")
        assert len(cards) == 0
