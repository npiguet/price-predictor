"""Filling a reminder template the way Forge fills it (T030, FR-017, FR-018).

Templates and detail strings are the real ones: the templates from Forge's
``Keyword`` enum as ``extract-keyword-definitions`` writes them, the details
from keyword lines in the converted corpus.
"""

from __future__ import annotations

import pytest

from effects.domain.keyword_formatting import (
    FORMATTERS,
    cost_text,
    format_reminder,
    java_format,
    strip_specifiers,
)

_WARD = (
    "Whenever this permanent becomes the target of a spell or ability an "
    "opponent controls, counter it unless that player %s."
)
_TYPECYCLING = (
    "%s, Discard this card: Search your library for %s, reveal it, put it into "
    "your hand, then shuffle."
)
_ENCHANT = "Target a %1$s as you cast this. This card enters attached to that %1$s."
_BUSHIDO = (
    "Whenever this creature blocks or becomes blocked, it gets +%1$d/+%1$d until "
    "end of turn."
)


class TestJavaFormat:
    def test_ordinary_specifiers_take_arguments_in_turn(self):
        assert java_format("%s and %d", ["a", "2"]) == "a and 2"

    def test_an_explicit_index_repeats_its_value(self):
        assert java_format("%1$s then %1$s", ["x"]) == "x then x"

    def test_a_specifier_with_no_argument_is_left_for_stripping(self):
        assert java_format("%s: %s", ["{2}"]) == "{2}: %s"

    def test_stripping_removes_every_form_and_the_space_it_leaves(self):
        assert strip_specifiers("pay %s, then %1$d and %2$s.") == "pay, then and."


class TestCostText:
    @pytest.mark.parametrize(
        ("cost", "text"),
        [
            ("2", "{2}"),
            ("1 R", "{1}{R}"),
            ("X 3 U", "{X}{3}{U}"),
            ("W/U", "{W/U}"),
            ("B Sac<1/Creature>", "{B}, Sacrifice a creature"),
            ("PayLife<3>", "Pay 3 life"),
            ("T", "{T}"),
        ],
    )
    def test_forge_cost_strings_render_as_toSimpleString_does(self, cost, text):
        assert cost_text(cost) == text


class TestFormatters:
    def test_ward_reads_pays_before_a_mana_cost(self):
        assert format_reminder("Ward", _WARD, "2").endswith("unless that player pays {2}.")

    def test_ward_rewords_a_life_payment(self):
        assert format_reminder("Ward", _WARD, "PayLife<3>").endswith(
            "unless that player pays 3 life."
        )

    def test_typecycling_names_a_card_of_its_type(self):
        filled = format_reminder("KeywordWithCostAndType", _TYPECYCLING, "Basic:1 B")
        assert filled.startswith("{1}{B}, Discard this card")
        assert "for a basic land card," in filled

    def test_enchant_fills_its_type_at_both_uses(self):
        filled = format_reminder("KeywordWithType", _ENCHANT, "Creature")
        assert filled.count("creature") == 2

    def test_an_amount_fills_every_explicit_use(self):
        assert "+1/+1" in format_reminder("KeywordWithAmount", _BUSHIDO, "1")

    def test_an_x_amount_reads_x(self):
        assert "+X/+X" in format_reminder("KeywordWithAmount", _BUSHIDO, "X")

    def test_an_unknown_formatter_fills_the_raw_values(self):
        assert format_reminder(None, "Equip %s to %s.", "3") == "Equip 3 to."

    def test_an_unparseable_detail_falls_back_to_the_generic_wording(self):
        filled = format_reminder("KeywordWithAmount", _BUSHIDO, "Sunburst")
        assert "%" not in filled

    def test_no_formatter_leaves_a_specifier(self):
        """Every class Forge's keyword table names has an entry."""
        for template, details in ((_WARD, "1"), (_ENCHANT, "Land"), (_BUSHIDO, "2")):
            for name in FORMATTERS:
                assert "%" not in format_reminder(name, template, details)
