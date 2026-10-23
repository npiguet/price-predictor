"""The one spelling of a keyword name (``domain/keyword_names.py``)."""

from __future__ import annotations

from effects.domain.keyword_names import normalize_keyword, overlay_keyword


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

    def test_any_other_keyword_is_its_normalized_name(self):
        assert overlay_keyword("Ward:2") == "ward"
        assert overlay_keyword("first strike") == "first_strike"
