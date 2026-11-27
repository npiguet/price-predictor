"""Resolving a line query and the nearest-neighbour check."""


from __future__ import annotations

import numpy as np

from effects.application.geometry_checks import (
    NEIGHBOUR_QUERIES,
    CachedVectors,
    check_nearest_neighbours,
    nearest_neighbours,
)
from effects.domain.evaluation_report import CheckStatus
from effects.domain.line_query import LineQuery
from tests.unit.effects.application.geometry_checks_support import (
    TAX_PROSE,
    _two_cards_one_prose,
)
from tests.unit.effects.selection_builders import read_cache


class TestResolvingAQuery:
    def test_a_query_resolves_by_prose_to_the_script_the_model_read(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        resolved = loaded.resolve(
            LineQuery("cardname deals 2 damage to any target."),
        )
        assert resolved.key == "DealDamage | NumDmg$ 2"

    def test_matching_ignores_case_and_whitespace_only(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        assert loaded.resolve(LineQuery("  FLYING ")).key == "Flying"
        assert loaded.resolve(LineQuery("flying.")).key is None
        assert loaded.resolve(LineQuery("fly")).key is None

    def test_an_absent_line_says_so(self, cache):
        cards, abilities = cache
        resolved = read_cache(abilities, cards, surface="script").resolve(
            LineQuery("draw a card."),
        )
        assert resolved.key is None
        assert "no line" in resolved.problem

    def test_a_prose_on_several_scripts_is_ambiguous_without_a_card(
        self, tmp_path,
    ):
        cards, abilities = _two_cards_one_prose(tmp_path)
        loaded = read_cache(abilities, cards, surface="script")
        ambiguous = loaded.resolve(LineQuery(TAX_PROSE))
        assert ambiguous.key is None
        assert "2 scripts" in ambiguous.problem
        assert loaded.resolve(LineQuery(TAX_PROSE, card="Two")).key == (
            "RaiseCost | Amount$ 2 | Type$ Spell"
        )

    def test_a_card_that_does_not_print_the_line_does_not_resolve(self, cache):
        cards, abilities = cache
        resolved = read_cache(abilities, cards, surface="script").resolve(
            LineQuery("flying", card="shock"),
        )
        assert resolved.key is None
        assert "shock" in resolved.problem

    def test_a_line_is_displayed_in_prose(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        assert loaded.label("DealDamage | NumDmg$ 2") == (
            "CARDNAME deals 2 damage to any target."
        )


class TestNearestNeighbours:
    def test_the_nearest_neighbour_is_the_most_similar_text(self):
        cached = CachedVectors(by_text={
            "a": np.array([1.0, 0.0]),
            "close": np.array([0.99, 0.14]),
            "far": np.array([0.0, 1.0]),
        })
        assert nearest_neighbours(cached, "a", k=1)[0][0] == "close"

    def test_the_query_is_not_its_own_neighbour(self):
        cached = CachedVectors(by_text={
            "a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
        })
        assert [t for t, _ in nearest_neighbours(cached, "a")] == ["b"]

    def test_an_unknown_query_has_no_neighbours(self):
        cached = CachedVectors(by_text={"a": np.array([1.0, 0.0])})
        assert nearest_neighbours(cached, "never encoded") == []

    def test_the_check_skips_on_an_empty_cache(self):
        result = check_nearest_neighbours(
            CachedVectors(), (LineQuery("draw a card."),),
        )
        assert result.status is CheckStatus.SKIPPED

    def test_the_check_reports_and_blocks_nothing(self, cache):
        cards, abilities = cache
        result = check_nearest_neighbours(
            read_cache(abilities, cards, surface="script"),
            (LineQuery("flying"),),
        )
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks

    def test_neighbours_are_listed_in_prose_for_the_reader(self, cache):
        cards, abilities = cache
        result = check_nearest_neighbours(
            read_cache(abilities, cards, surface="script"),
            (LineQuery("flying"),),
        )
        assert "CARDNAME deals 2 damage to any target." in result.detail
        assert "DealDamage" not in result.detail

    def test_a_query_that_resolves_to_nothing_is_named(self, cache):
        """A silently dropped query shrinks the comparison without saying so."""
        cards, abilities = cache
        result = check_nearest_neighbours(
            read_cache(abilities, cards, surface="script"),
            (LineQuery("flying"), LineQuery("draw a card.")),
        )
        assert "draw a card." in result.detail
        assert "unresolved" in result.detail
        assert result.values["unresolved_queries"] == 1

    def test_the_checked_in_queries_use_the_converters_spelling(self):
        for query in NEIGHBOUR_QUERIES:
            assert query.prose == query.prose.strip()
            assert query.prose.endswith(".")
            assert "this creature" not in query.prose
