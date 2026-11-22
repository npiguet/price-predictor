"""The state-based detectors skip the game-end record and no other effect half."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import sba_game_loss, sba_zone_move

_PREFIX = "3550376f-3c30-4231-90cd-cc0f1b308e87.1-mwqw0ogy54jx."
#: Two ward counter halves (a granted keyword, no ability line) and an engine effect's half.
_NOT_GAME_END = [_PREFIX + "7824", _PREFIX + "7839", _PREFIX + "11875"]


def _by_id(gen3_records, record_id):
    return next(r for r in gen3_records if r.record_id == record_id)


def _lethal_to_p1(record):
    """The real half, edited so P1 stands at 1 life and takes 1 damage."""
    state = replace(record.state, players=tuple(
        replace(p, life=1) if p.id == "P1" else p for p in record.state.players))
    hit = Event(type=EventType.DAMAGE_DEALT, subjects=("P1",), params={"amount": 1})
    return replace(record, state=state,
                   payload=replace(record.payload, events=(*record.payload.events, hit)))


@pytest.mark.parametrize("record_id", _NOT_GAME_END)
def test_a_half_that_is_not_the_game_end_record_is_read(gen3_records, record_id):
    record = _by_id(gen3_records, record_id)
    assert not record.is_game_end_record
    found = sba_game_loss.FAMILY.match(_lethal_to_p1(record), None, {"check": "life_zero"})
    assert found is not None and found.polarity is Polarity.FIRES


@pytest.mark.parametrize("record_id", _NOT_GAME_END)
def test_the_zone_move_detector_reads_that_half_too(gen3_records, record_id):
    """The half edited to deal 99 damage to a creature of its board."""
    record = _by_id(gen3_records, record_id)
    creature = next(e for e in record.state.entities
                    if e.on_battlefield and "creature" in e.types and e.pt is not None)
    hit = Event(type=EventType.DAMAGE_DEALT, subjects=(creature.id,), params={"amount": 99})
    edited = replace(record, payload=replace(record.payload, events=(*record.payload.events, hit)))
    found = sba_zone_move.FAMILY.match(edited, None, {"check": "lethal_damage"})
    assert found is not None and found.polarity is Polarity.FIRES


def test_the_game_end_records_are_still_skipped(gen3_records):
    ends = [r for r in gen3_records if r.is_game_end_record]
    assert ends
    for record in ends:
        edited = _lethal_to_p1(record)
        assert sba_game_loss.FAMILY.match(edited, None, {"check": "life_zero"}) is None
        for check in sba_zone_move.CHECKS:
            assert sba_zone_move.FAMILY.match(edited, None, {"check": check}) is None
