"""Tests for domain value objects."""

import pytest

from price_predictor.domain.value_objects import ManaCost


class TestManaCostParse:
    """Tests for ManaCost.parse() with Forge mana format."""

    def test_parse_two_generic_two_white(self):
        mc = ManaCost.parse("2 W W")
        assert mc is not None
        assert mc.total_mana_value == 4.0
        assert mc.generic_mana == 2
        assert mc.w == 2
        assert mc.u == 0
        assert mc.b == 0
        assert mc.r == 0
        assert mc.g == 0
        assert mc.color_count == 1
        assert mc.has_x is False

    def test_parse_single_red(self):
        mc = ManaCost.parse("R")
        assert mc is not None
        assert mc.total_mana_value == 1.0
        assert mc.generic_mana == 0
        assert mc.r == 1
        assert mc.color_count == 1

    def test_parse_generic_plus_two_colors(self):
        mc = ManaCost.parse("1 U R")
        assert mc is not None
        assert mc.total_mana_value == 3.0
        assert mc.generic_mana == 1
        assert mc.u == 1
        assert mc.r == 1
        assert mc.color_count == 2

    def test_parse_x_mana(self):
        mc = ManaCost.parse("X R")
        assert mc is not None
        assert mc.total_mana_value == 1.0
        assert mc.has_x is True
        assert mc.r == 1

    def test_parse_hybrid_mana(self):
        mc = ManaCost.parse("WU WU")
        assert mc is not None
        assert mc.has_hybrid is True
        assert mc.total_mana_value == 2.0
        assert mc.w >= 1
        assert mc.u >= 1

    def test_parse_phyrexian_mana(self):
        mc = ManaCost.parse("WP")
        assert mc is not None
        assert mc.has_phyrexian is True
        assert mc.total_mana_value == 1.0
        assert mc.w >= 1

    def test_parse_colorless_mana(self):
        mc = ManaCost.parse("C")
        assert mc is not None
        assert mc.total_mana_value == 1.0
        assert mc.colorless_mana == 1
        assert mc.generic_mana == 0
        assert mc.color_count == 0

    def test_parse_colorless_with_generic_and_color(self):
        """Matter Reshaper: 2 C — 2 generic + 1 colorless."""
        mc = ManaCost.parse("2 C")
        assert mc is not None
        assert mc.total_mana_value == 3.0
        assert mc.generic_mana == 2
        assert mc.colorless_mana == 1
        assert mc.color_count == 0

    def test_parse_snow_mana(self):
        mc = ManaCost.parse("S")
        assert mc is not None
        assert mc.total_mana_value == 1.0
        assert mc.color_count == 0

    def test_parse_no_cost_returns_none(self):
        result = ManaCost.parse("no cost")
        assert result is None

    def test_parse_empty_string_returns_none(self):
        result = ManaCost.parse("")
        assert result is None

    def test_parse_generic_only(self):
        mc = ManaCost.parse("3")
        assert mc is not None
        assert mc.total_mana_value == 3.0
        assert mc.generic_mana == 3
        assert mc.color_count == 0

    def test_parse_two_or_colored_hybrid(self):
        mc = ManaCost.parse("W2")
        assert mc is not None
        assert mc.has_hybrid is True
        assert mc.total_mana_value == 2.0
        assert mc.w >= 1

    def test_parse_zero_generic(self):
        mc = ManaCost.parse("0")
        assert mc is not None
        assert mc.total_mana_value == 0.0
        assert mc.generic_mana == 0

    def test_parse_double_x(self):
        mc = ManaCost.parse("X X")
        assert mc is not None
        assert mc.has_x is True
        assert mc.total_mana_value == 0.0

    def test_frozen(self):
        mc = ManaCost.parse("R")
        assert mc is not None
        with pytest.raises(AttributeError):
            mc.r = 5
