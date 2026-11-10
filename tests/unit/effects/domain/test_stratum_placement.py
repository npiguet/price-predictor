"""Global stratum placement and halves (FR-059, FR-068a, FR-070; research R3, E5)."""

from __future__ import annotations

import pytest

from effects.domain.corpus_curation import game_disjoint_draw
from effects.domain.corpus_plan import GameStrata
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleTag
from effects.domain.seeded_draw import stable_order
from effects.domain.strata import Half, Stratum
from effects.domain.stratum_placement import (
    GameCensus,
    PlacementRule,
    place_games,
    rule_test_eligible,
    rule_test_misses,
    rule_test_records_needed,
)
from tests.unit.effects.builders import make_record

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
GAMES = tuple(f"run.0-a.{i}" for i in range(600))
HELD = frozenset(GAMES[:12])


def ids_drawn(low: float, high: float, *, exclude=HELD) -> list[str]:
    return [g for g in GAMES if low <= game_disjoint_draw(g) < high and g not in exclude]


def rule(**overrides) -> PlacementRule:
    """Thresholds lowered so a few hundred games can meet them: one record of
    each polarity per half, two games per half."""
    values = dict(game_disjoint_share=0.01, keyword_share=0.15, card_disjoint_text_cap=0,
                  seed=3, min_games_per_half=2, max_item_error=1.0)
    values.update(overrides)
    return PlacementRule(**values)


def census(inventories, *, held=HELD, random_seat=(), text_games=None, keyword=()):
    return GameCensus(
        games=frozenset(GAMES), held_out=frozenset(held),
        random_seat=frozenset(random_seat), keyword_combat=frozenset(keyword),
        held_out_text_games=text_games if text_games is not None else {"t": set(held)},
        inventories=inventories,
    )


def alternating(rule_id: str, games=GAMES) -> dict[str, dict]:
    """``rule_id`` fires in every third game and does not in the next."""
    out: dict[str, dict] = {}
    for i, game in enumerate(games):
        if i % 3 < 2:
            out[game] = {(rule_id, FIRES if i % 3 == 0 else NOT): 1}
    return out


def polarities_in(strata, inventories, stratum, half, rule_id) -> set:
    return {
        polarity for game in strata.halves[stratum][half]
        for (found, polarity) in inventories.get(game, {}) if found == rule_id
    }


def test_the_default_test_record_target_is_hanley_mcneils_68():
    assert rule_test_records_needed() == 68


def test_both_polarities_land_in_both_halves_of_both_strata():
    inventories = alternating("704.5f")
    strata = place_games(census(inventories), rule())
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT):
        for half in Half:
            assert polarities_in(strata, inventories, stratum, half, "704.5f") == {FIRES, NOT}
    assert strata.left_out_of_half == {}


def test_a_rule_whose_records_cannot_fill_every_half_is_named():
    inventories = alternating("704.5f")
    lone = ids_drawn(0.0, 0.01)[0]
    inventories[lone] = {("702.2b", NOT): 1}
    for game in GAMES[100:200]:
        inventories.setdefault(game, {})[("702.2b", FIRES)] = 1
    strata = place_games(census(inventories), rule())
    named = strata.left_out_of_half
    assert set(named[Stratum.CARD_DISJOINT]) == set(Half)
    assert all("702.2b" in rules for rules in named[Stratum.CARD_DISJOINT].values())
    assert sum("702.2b" in rules for rules in named[Stratum.GAME_DISJOINT].values()) == 1
    assert all("704.5f" not in rules for by_half in named.values() for rules in by_half.values())


def test_a_random_seat_game_naming_a_held_out_card_is_card_disjoint():
    """FR-059 (T182): the card-disjoint stratum keeps every game naming a
    held-out card, random-seat games included; the random seat casts held-out
    cards the AI never casts."""
    seat_held = sorted(HELD)[:3]
    strata = place_games(census(alternating("704.5f"), random_seat=seat_held), rule())
    for game in seat_held:
        assert strata.stratum_of(game) is Stratum.CARD_DISJOINT
        assert strata.half_of(game) is not None


def test_the_game_disjoint_stratum_keeps_only_forge_ai_games():
    """FR-059: a random-seat game naming no held-out card trains, even one
    gen-2's draw would have placed."""
    seat_drawn = ids_drawn(0.0, 0.01)
    strata = place_games(census(alternating("704.5f"), random_seat=seat_drawn), rule())
    assert not set(seat_drawn) & strata.game_disjoint
    for game in seat_drawn:
        assert strata.stratum_of(game) is Stratum.TRAINING
        assert strata.half_of(game) is None


def test_a_short_cell_promotes_a_pool_game_and_never_one_outside_the_pool():
    pooled = ids_drawn(0.01, 0.1)[0]
    outside = ids_drawn(0.1, 1.0)[0]
    inventories = {pooled: {("y", FIRES): 1}, outside: {("z", FIRES): 1}}
    strata = place_games(census(inventories), rule())
    assert pooled in strata.game_disjoint
    assert strata.stratum_of(outside) is Stratum.TRAINING


def test_the_base_draw_is_gen_2s():
    base = set(ids_drawn(0.0, 0.01))
    keyword = ids_drawn(0.01, 0.15)[-1]
    strata = place_games(census({}, keyword=[keyword]), rule(min_games_per_half=0))
    assert strata.game_disjoint == base | {keyword}


def test_a_stratum_is_topped_up_to_two_halves_of_games():
    strata = place_games(census({}), rule(min_games_per_half=8))
    assert len(strata.game_disjoint) == 16
    pool = set(ids_drawn(0.0, 0.1))
    assert strata.game_disjoint <= pool
    halves = strata.halves[Stratum.GAME_DISJOINT]
    assert len(halves[Half.SELECTION]) == len(halves[Half.REPORTING]) == 8


def test_card_disjoint_admits_a_short_cells_game_first_under_the_text_cap():
    held = frozenset(GAMES[:4])
    last = stable_order(held, 3)[-1]
    inventories = {last: {("c", FIRES): 1}}
    strata = place_games(
        census(inventories, held=held), rule(card_disjoint_text_cap=2, min_games_per_half=0))
    assert len(strata.card_disjoint) == 2 and last in strata.card_disjoint


def test_a_held_out_game_resolving_no_held_out_text_is_dropped():
    held = frozenset(GAMES[:4])
    strata = place_games(census({}, held=held, text_games={"t": set(GAMES[:2])}), rule())
    assert strata.card_disjoint == frozenset(GAMES[:2])
    assert strata.stratum_of(GAMES[3]) is Stratum.DROPPED_HELD_OUT


def test_halves_balance_each_cell_and_the_game_counts():
    held = frozenset(GAMES[:8])
    inventories = {game: {("r", FIRES): 1} for game in GAMES[:4]}
    strata = place_games(census(inventories, held=held), rule(min_games_per_half=0))
    halves = strata.halves[Stratum.CARD_DISJOINT]
    for half in Half:
        assert len(halves[half]) == 4
        assert len(halves[half] & set(GAMES[:4])) == 2


def test_placement_is_a_function_of_the_corpus():
    inventories = alternating("704.5f")
    first = place_games(census(inventories), rule())
    reversed_inventories = dict(reversed(list(inventories.items())))
    assert place_games(census(reversed_inventories), rule()) == first
    assert place_games(census(inventories), rule(seed=4)) != first


def _tag(polarity=FIRES, observed=True):
    return RuleTag("704.5f", polarity, observed, ("E1",), ("E1",), "704.5f-a")


def test_rule_test_records_are_natural_observed_tags():
    assert rule_test_eligible(make_record(generation=3), _tag())
    assert not rule_test_eligible(make_record(generation=3, fork=True), _tag())
    assert not rule_test_eligible(make_record(generation=3, fork=True, interventional=True), _tag())
    assert not rule_test_eligible(make_record(generation=3, aimed_at=()), _tag())
    assert not rule_test_eligible(make_record(generation=3), _tag(Polarity.OTHER))
    assert not rule_test_eligible(make_record(generation=3), _tag(observed=False))


# ── what build-corpus can already tell of the precision target (E5) ─────


def _two_halves(selection: dict, reporting: dict):
    strata = GameStrata(
        card_disjoint=frozenset(), game_disjoint=frozenset({"a", "b"}),
        held_out=frozenset(),
        halves={Stratum.GAME_DISJOINT: {
            Half.SELECTION: frozenset({"a"}), Half.REPORTING: frozenset({"b"})}},
    )
    return strata, {"a": selection, "b": reporting}


def test_a_half_whose_rule_tests_meet_the_targets_is_no_miss():
    strata, inventories = _two_halves(
        {("r", FIRES): 68, ("r", NOT): 68}, {("r", FIRES): 68, ("r", NOT): 68})
    assert rule_test_misses(strata, inventories, rule(max_item_error=0.1,
                                                     max_component_error=0.1)) == []


def test_a_half_with_no_rule_to_test_misses_and_is_named():
    strata, inventories = _two_halves(
        {("r", FIRES): 68, ("r", NOT): 68}, {("r", FIRES): 3, ("r", NOT): 3})
    misses = rule_test_misses(strata, inventories, rule(max_item_error=0.1,
                                                        max_component_error=0.1))
    assert [(m.stratum, m.half, m.items) for m in misses] == [
        (Stratum.GAME_DISJOINT, Half.REPORTING, 0)]
    assert "game-disjoint reporting: rule tests" in misses[0].describe(0.05)


def test_one_rule_alone_misses_the_default_component_target():
    """√(SE²) / 1 = SE, so one rule at 0.0994 misses 0.05: a component needs
    several rules before it is that precise."""
    strata, inventories = _two_halves(
        {("r", FIRES): 68, ("r", NOT): 68}, {("r", FIRES): 68, ("r", NOT): 68})
    misses = rule_test_misses(strata, inventories, rule(max_item_error=0.1,
                                                        max_component_error=0.05))
    assert [m.items for m in misses] == [1, 1]
    assert misses[0].standard_error == pytest.approx(0.0994, abs=1e-4)
