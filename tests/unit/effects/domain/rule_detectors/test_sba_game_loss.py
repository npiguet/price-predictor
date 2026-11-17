"""``sba_game_loss`` on real gen-3 combats and effect halves."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import is_rescued_orphans
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
    pytest.fail("no combat in the fixture damages a player")


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
        FAMILY.validate({"check": "commander_damage"})


_BURN = "9442f17f-cb56-4893-8d4a-66b5538de180.3-mwhr1318yo3f.11670"


def _burn_kill(gen3_records):
    """A real effect half dealing 1 damage to P1, edited so P1 stands at 1
    life: the resolution takes P1 to 0 and holds no game-end event."""
    record = next(r for r in gen3_records if r.record_id == _BURN)
    assert not any(e.type in (EventType.PLAYER_WON, EventType.PLAYER_LOST)
                   for e in record.payload.listed_events())
    state = replace(record.state, players=tuple(
        replace(p, life=1) if p.id == "P1" else p for p in record.state.players))
    return replace(record, state=state)


def test_a_burn_kill_s_game_end_is_in_the_next_record_so_unreadable(gen3_records):
    """The resolution takes P1 to 0 life; the game-end rescue after it holds
    player_won. The kill fires, its outcome unreadable, and is no review."""
    found = FAMILY.match(_burn_kill(gen3_records), None, {"check": "life_zero"})
    assert found.polarity is Polarity.FIRES and found.observed is None
    assert found.entities == ("P1",)


def test_the_game_end_rescue_shows_no_side(gen3_records):
    """A real rescue record edited to hold the game end the collector files there."""
    rescue = next(r for r in gen3_records if is_rescued_orphans(r))
    won = Event(type=EventType.PLAYER_WON, subjects=("P0",))
    edited = replace(rescue, payload=replace(rescue.payload, events=(*rescue.payload.events, won)))
    assert any(e.type is EventType.PLAYER_WON for e in edited.payload.events)
    assert FAMILY.match(edited, None, {"check": "life_zero"}) is None


def test_a_player_both_dead_and_poisoned_is_neither_check_s_near_miss(by_kind):
    """Lethal damage with the ninth poison counter: the life loss ends the
    game, so poison's near miss cannot say what ended it. A real combat edited
    to the lethal hit."""
    record, player = _player_hit(by_kind)
    state = replace(record.state, players=tuple(
        replace(p, life=1, poison=9) if p.id == player else p for p in record.state.players))
    combat = replace(record, state=state)
    found = FAMILY.match(combat, None, {"check": "life_zero"})
    assert found.polarity is Polarity.FIRES and found.entities == (player,)
    assert FAMILY.match(combat, None, {"check": "poison_ten"}) is None
