"""``outcome_event``'s ``subject_in_combat`` on a player: 510.1b-1 reads damage to a player."""

from __future__ import annotations

import pytest

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.outcome_event import FAMILY

#: 510.1b-1's declaration.
UNBLOCKED = {"event": "damage_dealt", "combat": True, "subject_in_combat": False}


def _damage(record):
    return [e for e in record.payload.listed_events()
            if e.type.value == "damage_dealt" and (e.params or {}).get("combat")]


def _to_player(by_kind):
    return [r for r in by_kind("combat") if any(e.subjects[0].startswith("P") for e in _damage(r))]


def _unblocked_to_player(by_kind):
    """Combats whose every damage event to a player is an unblocked attacker's."""
    return [r for r in _to_player(by_kind)
            if all(not r.payload.blocks.get(e.params["source"]) for e in _damage(r)
                   if e.subjects[0].startswith("P"))]


def _to_creature_only(by_kind):
    return [r for r in by_kind("combat")
            if _damage(r) and not any(e.subjects[0].startswith("P") for e in _damage(r))]


def test_the_fixture_holds_both_kinds_of_combat(by_kind):
    assert _to_player(by_kind) and _to_creature_only(by_kind)


def test_an_unblocked_attacker_s_damage_to_the_player_it_attacks_fires_510_1b(by_kind):
    for record in _unblocked_to_player(by_kind):
        found = FAMILY.match(record, None, UNBLOCKED)
        assert found is not None and found.polarity is Polarity.FIRES, record.record_id


def test_the_outcome_names_the_defending_player(by_kind):
    record = _unblocked_to_player(by_kind)[0]
    assert "P0" in FAMILY.match(record, None, UNBLOCKED).outcome


def test_a_blocked_attacker_s_damage_to_its_blocker_does_not_fire_510_1b(by_kind):
    for record in _to_creature_only(by_kind):
        found = FAMILY.match(record, None, UNBLOCKED)
        assert found is None or found.polarity is not Polarity.FIRES, record.record_id


def test_a_player_no_attacker_attacks_reads_nothing(by_kind):
    """Damage to a player whom no attacker attacks is not an unblocked attacker's."""
    from dataclasses import replace
    record = _unblocked_to_player(by_kind)[0]
    entities = tuple(
        replace(e, combat=None) if e.combat is not None else e for e in record.state.entities)
    state = replace(record.state, entities=entities)
    found = FAMILY.match(replace(record, state=state), None, UNBLOCKED)
    assert found is None or found.polarity is not Polarity.FIRES


@pytest.mark.parametrize("param", ["subject_is_token", "controller_is_owner", "subject_type"])
def test_the_other_qualifiers_stay_unreadable_on_a_player(by_kind, param):
    value = {"subject_is_token": False, "controller_is_owner": True, "subject_type": "creature"}
    record = _to_player(by_kind)[0]
    found = FAMILY.match(record, None, {"event": "damage_dealt", "combat": True,
                                        param: value[param]})
    assert found is None


def _blocked(record):
    """The record's attackers all blocked, as a trampler's excess damage to the player
    reads (702.19b): the damage events themselves are untouched."""
    from dataclasses import replace
    blocks = {attacker: ("E-blocker",) for attacker in record.payload.attackers}
    return replace(record, payload=replace(record.payload, blocks=blocks))


def test_a_blocked_attacker_s_damage_to_the_player_does_not_fire_510_1b(by_kind):
    """CPC1 D-5, task FR4: the player is attacked, but the damage's source is blocked."""
    record = next(r for r in _to_player(by_kind)
                  if any(e.subjects[0].startswith("P") and (e.params or {}).get("source")
                         in r.payload.attackers for e in _damage(r)))
    assert FAMILY.match(record, None, UNBLOCKED).polarity is Polarity.FIRES
    found = FAMILY.match(_blocked(record), None, UNBLOCKED)
    assert found is None or found.polarity is not Polarity.FIRES


def test_a_real_blocked_trampler_s_excess_damage_does_not_fire_510_1b(by_kind):
    """The pilot's shape in a fixture record: E43, blocked by E9, deals 3 to the player."""
    blocked = [r for r in _to_player(by_kind) if r not in _unblocked_to_player(by_kind)]
    assert blocked
    for record in blocked:
        found = FAMILY.match(record, None, UNBLOCKED)
        assert found is None or found.polarity is not Polarity.FIRES, record.record_id
