"""``sba_game_loss`` on real gen-2 combats."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.sba_game_loss import FAMILY


def _player_hit(by_kind):
    """A real combat dealing damage to a player who survives it."""
    for record in by_kind("combat"):
        if record.fork:
            continue
        hit = {s for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
               for s in e.subjects if s.startswith("P")}
        if hit:
            return record, sorted(hit)[0]
    pytest.skip("no combat in the fixture damages a player")


def test_a_player_who_survives_the_hit_is_the_near_miss(by_kind):
    record, player = _player_hit(by_kind)
    found = FAMILY.match(record, None, {"check": "life_zero"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and player in found.entities


def test_the_same_combat_edited_to_a_lethal_hit_fires_and_ends_the_game(by_kind):
    record, player = _player_hit(by_kind)
    low = replace(record.state, players=tuple(
        replace(p, life=1) if p.id == player else p for p in record.state.players))
    winner = next(p.id for p in record.state.players if p.id != player)
    events = (*record.payload.events, Event(type=EventType.PLAYER_WON, subjects=(winner,)))
    edited = replace(record, state=low, payload=replace(record.payload, events=events))
    found = FAMILY.match(edited, None, {"check": "life_zero"})
    assert found.polarity is Polarity.FIRES and found.observed and player in found.entities


def test_poison_reads_its_own_counter(by_kind):
    record, player = _player_hit(by_kind)
    nine = replace(record.state, players=tuple(
        replace(p, poison=9) if p.id == player else p for p in record.state.players))
    events = (*record.payload.events,
              Event(type=EventType.POISON_CHANGE, subjects=(player,), params={"delta": 1}))
    edited = replace(record, state=nine, payload=replace(record.payload, events=events))
    assert FAMILY.match(edited, None, {"check": "poison_ten"}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"check": "poison_ten"}) is None


def test_an_unknown_check_is_refused(by_kind):
    with pytest.raises(ValueError, match="unknown check"):
        FAMILY.match(by_kind("combat")[0], None, {"check": "commander_damage"})
