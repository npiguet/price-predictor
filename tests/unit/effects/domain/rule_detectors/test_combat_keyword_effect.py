"""``combat_keyword_effect`` on real gen-3 combat records and on hand-built boards (025 T030).

The gen-3 fixture's combats hold vigilance and a first striker in the regular
step; toxic, absorb and trample over planeswalkers occur in no pilot record,
so their shapes are built here from the records' own spelling (a defender
rendered ``Name (id)``, a ``poison_change`` naming the attacker's controller,
a ``damage_prevented`` at the creature).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import families
from effects.domain.rule_detectors.combat_keyword_effect import FAMILY
from effects.domain.state_snapshot import (
    CombatStatus,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
)
from tests.unit.effects.builders import (
    GEN3_RECORDS,
    GEN3_SIDECAR_FOLDERS,
    make_entity,
    make_record,
    make_snapshot,
)


@pytest.fixture(scope="module")
def gen3_combats():
    from effects.infrastructure.shard_io import read_shard

    return [r for r in read_shard(GEN3_RECORDS) if r.kind is RecordKind.COMBAT]


@pytest.fixture(scope="module")
def gen3_sidecars():
    from effects.infrastructure.converted_trees import SourceTrees

    return SourceTrees.of(list(GEN3_SIDECAR_FOLDERS)).sidecar_cache()


# ── hand-built boards ───────────────────────────────────────────────────


def _creature(entity_id, keywords=(), *, controller="P0", power=3, tapped=False, combat=None):
    return make_entity(entity_id, entity_id.lower(), controller=controller, types=("creature",),
                       pt=PowerToughness((power, 3)), tapped=tapped, combat=combat,
                       granted_temporary=GrantedTemporary(keywords=tuple(keywords)))


def _damage(source, subject, amount=3):
    return Event(type=EventType.DAMAGE_DEALT, subjects=(subject,),
                 params={"amount": amount, "combat": True, "cause": source, "source": source})


def _combat(entities, events=(), substep="regular"):
    players = tuple(PlayerState(id=p, life=20, hand=3, library=30, graveyard=0)
                    for p in ("P0", "P1"))
    state = make_snapshot(entities=tuple(entities), players=players)
    state = replace(state, global_=replace(state.global_, combat_substep=substep))
    attackers = tuple(e.id for e in entities if e.combat and e.combat.attacking)
    return make_record(RecordKind.COMBAT, state=state,
                       payload=CombatPayload(attackers=attackers, events=tuple(events)))


def _attacking(blocked_by=()):
    return CombatStatus(attacking="p2", blocked_by=tuple(blocked_by))


def test_the_family_is_registered():
    assert families()["combat_keyword_effect"] is FAMILY


@pytest.mark.parametrize("keyword", ["toxic", "poisonous"])
def test_poison_fires_on_combat_damage_to_the_player(keyword):
    events = [_damage("E1", "P1"),
              Event(type=EventType.POISON_CHANGE, subjects=("P1",),
                    params={"delta": 2, "cause": "P0"})]
    found = FAMILY.match(_combat([_creature("E1", (keyword,), combat=_attacking())], events),
                         None, {"keyword": keyword})
    assert found.polarity is Polarity.FIRES
    assert (found.condition, found.outcome) == (("E1",), ("P1",))
    # Toxic's poison is in the step's record; poisonous's trigger resolves in another.
    assert found.observed is (True if keyword == "toxic" else None)


def test_toxic_without_its_poison_is_reviewed():
    found = FAMILY.match(_combat([_creature("E1", ("toxic",), combat=_attacking())],
                                 [_damage("E1", "P1")]), None, {"keyword": "toxic"})
    assert found.polarity is Polarity.FIRES and found.observed is False


def test_a_blocked_toxic_attacker_is_the_near_miss_at_the_defending_player():
    board = [_creature("E1", ("toxic",), combat=_attacking(("E2",))),
             _creature("E2", controller="P1", combat=CombatStatus(blocking=("E1",)))]
    found = FAMILY.match(_combat(board, [_damage("E1", "E2"), _damage("E2", "E1")]),
                         None, {"keyword": "toxic"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and found.observed is False
    assert (found.condition, found.outcome) == (("E1",), ("P1",))


def test_another_attacker_s_poison_leaves_the_near_miss_unreadable():
    board = [_creature("E1", ("toxic",), combat=_attacking(("E2",))),
             _creature("E2", controller="P1", combat=CombatStatus(blocking=("E1",))),
             _creature("E3", ("infect",), combat=_attacking())]
    events = [_damage("E3", "P1"), Event(type=EventType.POISON_CHANGE, subjects=("P1",),
                                         params={"delta": 3, "cause": "P0"})]
    found = FAMILY.match(_combat(board, events), None, {"keyword": "toxic"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and found.observed is None


def test_vigilance_sides_are_carriers_against_other_attackers():
    vigilant = _creature("E1", ("vigilance",), combat=_attacking())
    plain = _creature("E2", tapped=True, combat=_attacking())
    found = FAMILY.match(_combat([vigilant, plain]), None, {"keyword": "vigilance"})
    assert (found.polarity, found.observed) == (Polarity.FIRES, True)
    assert (found.condition, found.outcome) == (("E1",), ("E1",))
    alone = FAMILY.match(_combat([plain]), None, {"keyword": "vigilance"})
    assert (alone.polarity, alone.observed, alone.condition) == (
        Polarity.DOES_NOT_FIRE, False, ("E2",))


def test_vigilance_absent_puts_the_tapped_attacker_on_the_fires_side():
    """508.1f: an attacker without vigilance is tapped by attacking."""
    plain = _creature("E2", tapped=True, combat=_attacking())
    found = FAMILY.match(_combat([plain]), None,
                         {"keyword": "vigilance", "fires_when": "absent"})
    assert (found.polarity, found.observed, found.condition) == (Polarity.FIRES, True, ("E2",))
    vigilant = FAMILY.match(_combat([_creature("E1", ("vigilance",), combat=_attacking())]),
                            None, {"keyword": "vigilance", "fires_when": "absent"})
    assert (vigilant.polarity, vigilant.observed) == (Polarity.DOES_NOT_FIRE, False)


def test_a_tapped_vigilant_attacker_still_deals_damage():
    """506.4b: tapped after the declaration, it stays attacking."""
    tapped = _creature("E1", ("vigilance",), tapped=True, combat=_attacking())
    found = FAMILY.match(_combat([tapped], [_damage("E1", "P1")]), None,
                         {"keyword": "vigilance", "tapped": True})
    assert (found.polarity, found.observed, found.outcome) == (Polarity.FIRES, True, ("P1",))
    untapped = _creature("E1", ("vigilance",), combat=_attacking())
    assert FAMILY.match(_combat([untapped], [_damage("E1", "P1")]), None,
                        {"keyword": "vigilance", "tapped": True}) is None


def _walker(loyalty=2, zone="battlefield"):
    return make_entity("E9", "walker", controller="P1", types=("planeswalker",), zone=zone,
                       counters={"LOYALTY": loyalty})


def _at_walker(entity_id, keywords, defender="walker (9)"):
    return _creature(entity_id, keywords, combat=CombatStatus(attacking=defender))


def test_trample_over_planeswalkers_assigns_the_excess_to_the_controller():
    events = [_damage("E1", "E9", 2), _damage("E1", "P1", 1)]
    found = FAMILY.match(_combat([_at_walker("E1", ("trample_over_planeswalkers",)), _walker()],
                                 events), None, {"keyword": "trample_over_planeswalkers"})
    assert (found.polarity, found.observed) == (Polarity.FIRES, True)
    assert (found.condition, found.outcome) == (("E1",), ("P1",))


def test_plain_trample_at_a_planeswalker_is_the_near_miss():
    found = FAMILY.match(_combat([_at_walker("E1", ("trample",)), _walker()],
                                 [_damage("E1", "E9", 3)]),
                         None, {"keyword": "trample_over_planeswalkers"})
    assert (found.polarity, found.observed, found.outcome) == (
        Polarity.DOES_NOT_FIRE, False, ("P1",))


def test_no_excess_over_loyalty_shows_nothing():
    found = FAMILY.match(_combat([_at_walker("E1", ("trample_over_planeswalkers",)),
                                  _walker(loyalty=5)]),
                         None, {"keyword": "trample_over_planeswalkers"})
    assert found is None


def test_a_gone_planeswalker_flips_with_fires_when_absent():
    """702.19f: without the keyword, nothing reaches the player once the planeswalker is gone."""
    board = [_at_walker("E1", (), defender="<Nothing>")]
    params = {"keyword": "trample_over_planeswalkers", "defender": "gone"}
    carried = FAMILY.match(_combat(board), None, params)
    assert (carried.polarity, carried.observed) == (Polarity.DOES_NOT_FIRE, False)
    absent = FAMILY.match(_combat(board), None, {**params, "fires_when": "absent"})
    assert (absent.polarity, absent.observed, absent.outcome) == (Polarity.FIRES, True, ("P1",))
    graveyard = [_at_walker("E1", ("trample_over_planeswalkers",)), _walker(zone="graveyard")]
    found = FAMILY.match(_combat(graveyard, [_damage("E1", "P1")]), None, params)
    assert (found.polarity, found.observed) == (Polarity.FIRES, True)


def test_absorb_fires_where_damage_aimed_at_the_carrier_is_prevented():
    board = [_creature("E1", combat=_attacking(("E2",))),
             _creature("E2", ("absorb",), controller="P1", combat=CombatStatus(blocking=("E1",)))]
    prevented = Event(type=EventType.DAMAGE_PREVENTED, subjects=("E2",),
                      params={"amount": 1, "source": "E1"})
    found = FAMILY.match(_combat(board, [_damage("E1", "E2", 2), prevented]),
                         None, {"keyword": "absorb"})
    assert (found.polarity, found.observed) == (Polarity.FIRES, True)
    assert (found.condition, found.outcome) == (("E2",), ("E2",))
    unprevented = FAMILY.match(_combat(board, [_damage("E1", "E2", 3)]), None,
                               {"keyword": "absorb"})
    assert unprevented.observed is False
    assert FAMILY.match(_combat(board), None, {"keyword": "absorb"}) is None


def test_a_first_striker_dealing_regular_damage_gained_it_late():
    """702.7c: it had no first strike when the first-strike step began."""
    striker = _creature("E1", ("first_strike",), combat=_attacking())
    found = FAMILY.match(_combat([striker], [_damage("E1", "P1")]), None,
                         {"keyword": "first_strike"})
    assert (found.polarity, found.observed, found.outcome) == (Polarity.FIRES, True, ("P1",))
    quiet = FAMILY.match(_combat([striker]), None, {"keyword": "first_strike"})
    assert (quiet.polarity, quiet.observed, quiet.outcome) == (
        Polarity.DOES_NOT_FIRE, False, ("P1",))
    assert FAMILY.match(_combat([striker], substep="first_strike"), None,
                        {"keyword": "first_strike"}) is None
    double = _creature("E1", ("first_strike", "double_strike"), combat=_attacking())
    assert FAMILY.match(_combat([double], [_damage("E1", "P1")]), None,
                        {"keyword": "first_strike"}) is None


@pytest.mark.parametrize("params", [
    {"keyword": "flying"},
    {"keyword": "trample"},
    {"keyword": "toxic", "fires_when": "absent"},
    {"keyword": "vigilance", "defender": "gone"},
    {"keyword": "trample_over_planeswalkers", "defender": "battle"},
    {"keyword": "absorb", "tapped": True},
])
def test_parameters_outside_the_family_are_refused(params):
    with pytest.raises(ValueError):
        FAMILY.match(_combat([]), None, params)


def test_a_probe_shows_nothing():
    record = _combat([_creature("E1", ("vigilance",), combat=_attacking())])
    probe = replace(record, fork=True, payload=replace(record.payload, probed_keyword="trample",
                                                       probed_entity="E1"))
    assert probe.is_probe and FAMILY.match(probe, None, {"keyword": "vigilance"}) is None


# ── real gen-3 combats ──────────────────────────────────────────────────


def test_real_vigilant_attackers_stay_untapped(gen3_combats, gen3_sidecars):
    found = [FAMILY.match(r, gen3_sidecars, {"keyword": "vigilance"}) for r in gen3_combats]
    fires = [f for f in found if f is not None and f.polarity is Polarity.FIRES]
    near = [f for f in found if f is not None and f.polarity is Polarity.DOES_NOT_FIRE]
    if not fires or not near:
        pytest.skip("the fixture holds no vigilant attacker or no other attacker")
    assert all(f.observed for f in fires)
    assert sum(f.observed is False for f in near) > len(near) * 0.9
    assert all(f.condition == f.outcome for f in fires)


def test_real_regular_step_first_strikers_are_the_near_miss(gen3_combats, gen3_sidecars):
    found = [f for r in gen3_combats
             if (f := FAMILY.match(r, gen3_sidecars, {"keyword": "first_strike"})) is not None]
    if not found:
        pytest.skip("the fixture holds no first striker in a regular step")
    for f in found:
        assert f.observed is (f.polarity is Polarity.FIRES)
        assert f.condition and f.outcome
