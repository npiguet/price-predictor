"""Tests for the consolidated card taxonomy registries."""

from __future__ import annotations

from price_predictor.domain.card_taxonomy import (
    CARD_TYPES,
    KNOWN_CARD_TYPES,
    KNOWN_SUPERTYPES,
    LAYOUTS,
    RARITIES,
    SUPERTYPES,
    VALID_LAYOUTS,
    VALID_RARITIES,
)


class TestRegistries:
    def test_card_types_includes_tribal_and_kindred(self) -> None:
        """Tribal and Kindred are first-class card types in modern MTGJSON."""
        assert "Tribal" in CARD_TYPES
        assert "Kindred" in CARD_TYPES

    def test_card_types_starts_with_creature(self) -> None:
        """Encoder one-hot positions are stable: Creature must remain at index 0."""
        assert CARD_TYPES[0] == "Creature"

    def test_known_card_types_matches_card_types(self) -> None:
        assert KNOWN_CARD_TYPES == frozenset(CARD_TYPES)

    def test_known_supertypes_matches_supertypes(self) -> None:
        assert KNOWN_SUPERTYPES == frozenset(SUPERTYPES)

    def test_layouts_in_valid_layouts(self) -> None:
        assert VALID_LAYOUTS == frozenset(LAYOUTS)

    def test_valid_rarities_includes_special_and_bonus(self) -> None:
        """The validation set is broader than the encoded set so MTGJSON's
        special/bonus rarities can be stored without crashing."""
        assert "special" in VALID_RARITIES
        assert "bonus" in VALID_RARITIES
        for r in RARITIES:
            assert r in VALID_RARITIES

    def test_rarities_canonical_four(self) -> None:
        assert RARITIES == ("common", "uncommon", "rare", "mythic")
