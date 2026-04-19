"""Unit tests for the basic-land distribution domain function."""

from __future__ import annotations

from price_predictor.domain.card_text import ConvertedCardText
from sealed.domain.manabase import compute_basic_lands


def _ct(text: str) -> ConvertedCardText:
    return ConvertedCardText(text)


class TestBasicLandComputation:
    def test_fills_to_40_total(self):
        """Basic lands should fill remaining slots to 40 total cards."""
        nonland_texts = [
            _ct(f"name: card_{i}\nmana cost: {{R}}\ntypes: creature") for i in range(23)
        ]
        lands = compute_basic_lands(nonland_texts)
        assert sum(lands.values()) == 17

    def test_counts_duplicate_card_slots(self):
        """Duplicate card slots must each contribute to n_basics and pip counts."""
        text = _ct("name: card\nmana cost: {R}\ntypes: creature")
        nonland_texts = [text] * 23
        lands = compute_basic_lands(nonland_texts)
        assert sum(lands.values()) == 17
        assert lands.get("Mountain", 0) == 17

    def test_proportional_to_color_pips(self):
        nonland_texts = [
            _ct(f"name: card_{i}\nmana cost: {{R}}\ntypes: creature") for i in range(23)
        ]
        lands = compute_basic_lands(nonland_texts)
        assert lands.get("Mountain", 0) == 17

    def test_multicolor_distributes_proportionally(self):
        texts = []
        for i in range(12):
            texts.append(_ct(f"name: red_{i}\nmana cost: {{R}}\ntypes: creature"))
        for i in range(11):
            texts.append(_ct(f"name: green_{i}\nmana cost: {{G}}\ntypes: creature"))
        lands = compute_basic_lands(texts)
        assert sum(lands.values()) == 17
        assert "Mountain" in lands
        assert "Forest" in lands

    def test_min_two_per_active_color(self):
        """A splash color with low pip count still gets at least 2 basics."""
        texts = [
            _ct(f"name: w_{i}\nmana cost: {{W}}\ntypes: creature") for i in range(22)
        ]
        texts.append(_ct("name: splash\nmana cost: {1}{B}\ntypes: creature"))
        lands = compute_basic_lands(texts)
        assert sum(lands.values()) == 17
        assert lands.get("Swamp", 0) >= 2
        assert lands.get("Plains", 0) >= 2

    def test_colorless_deck_fans_out(self):
        """A deck with no colored pips falls back to an even five-color split."""
        texts = [
            _ct(f"name: c_{i}\nmana cost: {{2}}\ntypes: artifact") for i in range(23)
        ]
        lands = compute_basic_lands(texts)
        assert sum(lands.values()) == 17
        # All five basic land types should appear at least twice (min-per-color rule)
        for basic in ("Plains", "Island", "Swamp", "Mountain", "Forest"):
            assert lands.get(basic, 0) >= 2
