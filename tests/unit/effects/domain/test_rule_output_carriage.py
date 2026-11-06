"""``rule_output_carriage``: which of a rule's map outputs a record carries for its tag."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload, RecordKind, ResolutionPayload, TriggerPayload
from effects.domain.rule_output_carriage import carried_outputs
from effects.domain.state_snapshot import PlayerState
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

PLAYERS = tuple(PlayerState(id=p, life=20, hand=3, library=30, graveyard=0) for p in ("P0", "P1"))


def _combat():
    """E2 dies to damage; E1 (the attacker) is untouched, P0 gains life."""
    board = (make_entity("E1", "attacker"), make_entity("E2", "blocker", controller="P1"))
    return make_record(RecordKind.COMBAT, generation=3,
                       state=make_snapshot(entities=board, players=PLAYERS),
                       payload=CombatPayload(attackers=("E1",), events=(
                           Event(EventType.DAMAGE_DEALT, ("E2",), {"amount": 2}),
                           Event(EventType.ZONE_CHANGE, ("E2",),
                                 {"from_zone": "battlefield", "to_zone": "graveyard"}),
                           Event(EventType.LIFE_CHANGE, ("P0",), {"delta": 2}))))


def test_a_per_entity_output_is_carried_only_at_an_affected_outcome_entity():
    record = _combat()
    outputs = ("zone_outcome", "damage_taken", "life_delta", "gate")
    assert carried_outputs(record, ("E2",), outputs) == {"zone_outcome", "damage_taken", "gate"}
    assert carried_outputs(record, ("P0",), outputs) == {"life_delta", "gate"}
    # The unaffected attacker carries nothing but the gate, which trains everywhere.
    assert carried_outputs(record, ("E1",), outputs) == {"gate"}
    assert carried_outputs(record, (), outputs) == frozenset()


def test_a_trigger_record_carries_its_fired_bit_and_no_per_entity_output():
    board = (make_entity("E1", "source"),)
    record = make_record(RecordKind.TRIGGER, generation=3,
                         state=make_snapshot(entities=board, players=PLAYERS),
                         payload=TriggerPayload(
                             event=Event(EventType.LIFE_CHANGE, ("P0",), {"delta": -1}),
                             fired=True))
    assert carried_outputs(record, ("P0", "E1"), ("trigger_fired", "life_delta")) == {
        "trigger_fired"}


def test_created_objects_and_an_unknown_name_read_at_the_record():
    record = make_record(generation=3, state=make_snapshot(players=PLAYERS),
                         payload=ResolutionPayload(events=(
                             Event(EventType.TOKEN_CREATED, ("P0",), {"count": 1}),)))
    assert carried_outputs(record, (), ("created_objects", "planned:thing")) == {
        "created_objects"}
