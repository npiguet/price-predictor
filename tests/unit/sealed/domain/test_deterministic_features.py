"""Unit tests for deterministic feature parsing from converted card text."""

from __future__ import annotations

import numpy as np

from price_predictor.domain.card_text import ConvertedCardText
from sealed.domain.deterministic_features import parse_deterministic_features


class TestManaCostBreakdown:
    def test_simple_red_spell(self):
        """Lightning Bolt: {R} → R=1, generic=0, mv=1."""
        text = "name: lightning bolt\nmana cost: {R}\ntypes: instant\nspell[1]: deal 3 damage."
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[4] == 1.0   # R pip count (W=1,U=2,B=3,R=4,G=5)
        assert feats[7] == 0.0   # generic mana
        assert feats[9] == 1.0   # mana value

    def test_multi_pip_spell(self):
        """Balduvian Hydra: {X}{R}{R} → R=2, X=1, generic=0, mv=2 (X=0 in mv)."""
        text = ("name: balduvian hydra\nmana cost: {X}{R}{R}\n"
                "types: creature hydra\npower toughness: 0/1")
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[4] == 2.0   # R pips
        assert feats[8] == 1.0   # X count
        assert feats[9] == 2.0   # mana value (X counts as 0)

    def test_generic_plus_colored(self):
        """Lab Brute: {3}{U} → U=1, generic=3, mv=4."""
        text = ("name: laboratory brute\nmana cost: {3}{U}\n"
                "types: creature zombie horror\npower toughness: 3/3")
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[2] == 1.0   # U pip
        assert feats[7] == 3.0   # generic
        assert feats[9] == 4.0   # mv

    def test_multicolor_spell(self):
        """{2}{R}{R} → R=2, generic=2, mv=4."""
        text = "name: test card\nmana cost: {2}{R}{R}\ntypes: creature"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[4] == 2.0   # R
        assert feats[7] == 2.0   # generic
        assert feats[9] == 4.0   # mv

    def test_colorless_pip(self):
        """{2}{C} → C=1, generic=2, mv=3."""
        text = "name: test\nmana cost: {2}{C}\ntypes: creature"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[6] == 1.0   # C (colorless pip)
        assert feats[7] == 2.0   # generic
        assert feats[9] == 3.0   # mv

    def test_no_mana_cost(self):
        """Card with no mana cost line → all mana features = 0."""
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        # Pips W/U/B/R/G/C/generic/X all zero
        for i in range(1, 9):
            assert feats[i] == 0.0
        assert feats[9] == 0.0  # mv


class TestLandDetection:
    def test_basic_land(self):
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[0] == 1.0  # is_land

    def test_nonbasic_land(self):
        text = "name: breeding pool\ntypes: land forest island\nactivated[1]: {T}: add {U} or {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[0] == 1.0

    def test_not_land(self):
        text = "name: bolt\nmana cost: {R}\ntypes: instant"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[0] == 0.0


class TestDevoidHandling:
    def test_devoid_overrides_color_flags(self):
        """Barrage Tyrant: {4}{R}, devoid → color flags all 0, is_colorless=1."""
        text = (
            "name: barrage tyrant\nmana cost: {4}{R}\ntypes: creature eldrazi\n"
            "power toughness: 5/3\nstatic: devoid"
        )
        feats = parse_deterministic_features(ConvertedCardText(text))
        # R pip still counted
        assert feats[4] == 1.0   # R pip (index 4)
        # But color flags zeroed by devoid (indices 10-14)
        for i in range(10, 15):
            assert feats[i] == 0.0, f"Color flag at index {i} should be 0 with devoid"
        assert feats[15] == 1.0  # is_colorless

    def test_non_devoid_sets_color_flags(self):
        text = "name: test\nmana cost: {R}{G}\ntypes: creature"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[13] == 1.0  # is_red (color_flag for R)
        assert feats[14] == 1.0  # is_green (color_flag for G)
        assert feats[15] == 0.0  # is_colorless


class TestManaProduction:
    def test_basic_land_produces_mana(self):
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[20] == 1.0  # produces G

    def test_dual_land_produces_two_colors(self):
        text = "name: breeding pool\ntypes: land forest island\nactivated[1]: {T}: add {U} or {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[17] == 1.0  # produces U
        assert feats[20] == 1.0  # produces G

    def test_creature_with_mana_ability(self):
        text = (
            "name: llanowar elves\nmana cost: {G}\ntypes: creature elf druid\n"
            "power toughness: 1/1\nactivated[1]: {T}: add {G}."
        )
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[20] == 1.0  # produces G

    def test_non_activated_add_not_counted(self):
        """Only activated abilities with 'add' are counted for mana production."""
        text = "name: test\nmana cost: {G}\ntypes: creature\ntriggered: add {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[20] == 0.0  # NOT counted — triggered, not activated

    def test_colorless_mana_production(self):
        text = "name: test\ntypes: land\nactivated[1]: {T}: add {C}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[21] == 1.0  # produces C


class TestPowerToughnessLoyalty:
    def test_creature_stats(self):
        text = "name: test\nmana cost: {G}\ntypes: creature\npower toughness: 3/3"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[23] == 3.0  # power
        assert feats[24] == 3.0  # toughness

    def test_star_power_is_zero(self):
        text = "name: test\nmana cost: {G}\ntypes: creature\npower toughness: */4"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[23] == 0.0  # * → 0
        assert feats[24] == 4.0

    def test_x_toughness_is_zero(self):
        text = "name: test\nmana cost: {G}\ntypes: creature\npower toughness: 2/X"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[23] == 2.0
        assert feats[24] == 0.0  # X → 0

    def test_planeswalker_loyalty(self):
        text = (
            "name: liliana\nmana cost: {3}{B}{B}\ntypes: legendary planeswalker liliana\n"
            "loyalty: 5"
        )
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[25] == 5.0  # loyalty

    def test_non_planeswalker_loyalty_zero(self):
        text = "name: test\nmana cost: {G}\ntypes: creature\npower toughness: 2/2"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[25] == 0.0

    def test_no_pt_line(self):
        text = "name: bolt\nmana cost: {R}\ntypes: instant"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[23] == 0.0
        assert feats[24] == 0.0


class TestZeroPadding:
    def test_indices_26_to_31_are_zero(self):
        text = "name: test\nmana cost: {R}\ntypes: creature\npower toughness: 2/2"
        feats = parse_deterministic_features(ConvertedCardText(text))
        for i in range(26, 32):
            assert feats[i] == 0.0, f"Index {i} should be zero padding"


class TestOutputShape:
    def test_returns_32_element_float_array(self):
        text = "name: test\nmana cost: {R}\ntypes: creature"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (32,)
        assert feats.dtype == np.float32


class TestManaCount:
    def test_single_mana_production(self):
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[22] == 1.0  # mana_count

    def test_multi_mana_production(self):
        """add {U} or {G} → mana_count = 1 (produces one mana per activation)."""
        text = "name: test\ntypes: land\nactivated[1]: {T}: add {U} or {G}"
        feats = parse_deterministic_features(ConvertedCardText(text))
        assert feats[22] == 1.0  # mana_count (one mana per activation)
