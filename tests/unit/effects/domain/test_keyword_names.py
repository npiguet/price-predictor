"""The one spelling of a keyword name (``domain/keyword_names.py``)."""

from __future__ import annotations

from effects.domain.keyword_names import (
    changed_keyword,
    normalize_keyword,
    overlay_keyword,
    with_implied,
)


class TestKeywordSpelling:
    """Forge writes what the card prints; this package spells them with
    underscores. The two-word keywords are where that bites: first strike and
    double strike are exactly the pair gate 2 counted zero observations of, on
    a corpus holding hundreds of them.
    """

    def test_a_two_word_keyword_gains_its_underscore(self):
        assert normalize_keyword("first strike") == "first_strike"
        assert normalize_keyword("double strike") == "double_strike"

    def test_a_one_word_keyword_is_unchanged(self):
        assert normalize_keyword("trample") == "trample"

    def test_a_parameter_is_dropped(self):
        """The keyword is the same keyword whatever its value, and the value
        reaches the model through the ability line rather than the overlay."""
        assert normalize_keyword("ward:2") == "ward"
        assert normalize_keyword("landwalk:forest") == "landwalk"
        assert normalize_keyword("Equip:1:Detective.YouCtrl:Detective") == "equip"

    def test_a_protection_keeps_its_quality_in_either_spelling(self):
        """What a protection protects from is which keyword it is: Forge's
        colon form reads as the printed form does."""
        assert normalize_keyword("Protection from blue") == "protection_from_blue"
        assert normalize_keyword("Protection:Goblin") == "protection_from_goblin"
        assert normalize_keyword("Protection:Card.MultiColor:multicolored") == (
            "protection_from_multicolored")
        assert normalize_keyword("Protection:Creature.Legendary:legendary creatures") == (
            "protection_from_legendary_creatures")

    def test_a_hexproof_from_a_quality_is_no_plain_hexproof(self):
        """Hexproof from black leaves the card a legal target of every other
        colour's spells, so it never reads as hexproof."""
        assert normalize_keyword("Hexproof") == "hexproof"
        assert normalize_keyword("Hexproof:Black") == "hexproof_from_black"
        assert normalize_keyword("hexproof:card.black:black") == "hexproof_from_black"
        assert normalize_keyword("Hexproof:Card.MonoColor:monocolored") == (
            "hexproof_from_monocolored")

    def test_a_selector_s_type_prefix_is_no_part_of_the_quality(self):
        assert normalize_keyword("Protection:Creature.Goblin") == "protection_from_goblin"
        assert normalize_keyword("Hexproof:Card.Black") == "hexproof_from_black"

    def test_a_quality_spelling_is_a_fixed_point(self):
        for raw in ("Protection:Goblin", "Hexproof:Black", "Protection from red"):
            assert normalize_keyword(normalize_keyword(raw)) == normalize_keyword(raw)

    def test_trample_over_planeswalkers_is_a_keyword_of_its_own(self):
        """702.19c: what follows Forge's colon changes what the trample does, as
        a protection's quality does; the printed form reads the same."""
        assert normalize_keyword("Trample:Planeswalker") == "trample_over_planeswalkers"
        assert normalize_keyword("trample:planeswalker") == "trample_over_planeswalkers"
        assert normalize_keyword("Trample over planeswalkers") == "trample_over_planeswalkers"
        assert normalize_keyword("trample_over_planeswalkers") == "trample_over_planeswalkers"
        assert normalize_keyword("Trample") == "trample"

    def test_case_and_padding_do_not_matter(self):
        assert normalize_keyword("  First Strike  ") == "first_strike"

    def test_every_damage_step_keyword_is_a_fixed_point(self):
        """Gate 2's table is already written in the normalized spelling, so
        normalizing it again must not move it."""
        from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS

        for row in DAMAGE_STEP_KEYWORDS:
            assert normalize_keyword(row.keyword) == row.keyword

    def test_forges_spelling_of_each_reaches_the_table(self):
        from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME

        for printed in ("First Strike", "Double Strike", "Deathtouch"):
            assert normalize_keyword(printed) in KEYWORDS_BY_NAME


class TestOverlaySpelling:
    def test_every_protection_is_the_one_protection_slot(self):
        assert overlay_keyword("protection from blue") == "protection"
        assert overlay_keyword("Protection from creatures") == "protection"

    def test_a_colon_form_protection_is_the_protection_slot_too(self):
        assert overlay_keyword("Protection:Goblin") == "protection"

    def test_a_hexproof_from_a_quality_is_not_the_hexproof_slot(self):
        assert overlay_keyword("Hexproof") == "hexproof"
        assert overlay_keyword("Hexproof:Black") != "hexproof"

    def test_trample_over_planeswalkers_is_the_trample_slot(self):
        """A creature with trample over planeswalkers has trample (702.19c), as
        Forge's ``hasKeyword(Keyword.TRAMPLE)`` reads it: the input's flag and
        the keywords-changed outputs are trample's."""
        assert overlay_keyword("Trample:Planeswalker") == "trample"
        assert changed_keyword("Trample:Planeswalker") == "trample"

    def test_any_other_keyword_is_its_normalized_name(self):
        assert overlay_keyword("Ward:2") == "ward"
        assert overlay_keyword("first strike") == "first_strike"


class TestChangedKeywordSpelling:
    """The keywords-gained and -lost outputs fold a hexproof from a quality
    onto ``hexproof`` as every protection folds onto ``protection``; the
    input's flag for full hexproof does not."""

    def test_a_hexproof_from_a_quality_is_a_hexproof_change(self):
        assert changed_keyword("Hexproof:Black") == "hexproof"
        assert overlay_keyword("Hexproof:Black") == "hexproof_from_black"

    def test_every_other_keyword_reads_as_the_overlay_does(self):
        for raw in ("Protection:Goblin", "Ward:2", "first strike", "Hexproof"):
            assert changed_keyword(raw) == overlay_keyword(raw)



class TestImpliedKeywords:
    def test_trample_over_planeswalkers_carries_trample_too(self):
        assert with_implied({"trample_over_planeswalkers"}) == {
            "trample_over_planeswalkers", "trample"}

    def test_other_keywords_imply_nothing(self):
        assert with_implied({"trample", "protection_from_red"}) == {
            "trample", "protection_from_red"}
