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


def _record(gen2_records, record_id):
    return next(r for r in gen2_records if r.record_id == record_id)


def test_a_burn_kill_s_game_end_is_in_the_next_record_so_unreadable(gen2_records):
    """The resolution takes P1 to 0 life; the game-end rescue after it holds
    player_won. The kill fires, its outcome unreadable, and is no review."""
    kill = _record(gen2_records, "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6392")
    found = FAMILY.match(kill, None, {"check": "life_zero"})
    assert found.polarity is Polarity.FIRES and found.observed is None
    assert found.entities == ("P1",)


def test_the_game_end_rescue_shows_no_side(gen2_records):
    rescue = _record(gen2_records, "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6393")
    assert any(e.type is EventType.PLAYER_WON for e in rescue.payload.events)
    assert FAMILY.match(rescue, None, {"check": "life_zero"}) is None


def test_a_player_both_dead_and_poisoned_is_neither_check_s_near_miss(gen2_records):
    """Twelve damage at 3 life and the ninth poison counter: the life loss
    ends the game, so poison's near miss cannot say what ended it."""
    combat = _record(gen2_records, "71253f20-5472-4ddd-8c58-27c78651878b.0-mv46oleh5eqw.8786")
    found = FAMILY.match(combat, None, {"check": "life_zero"})
    assert found.polarity is Polarity.FIRES and found.observed and found.entities == ("P0",)
    assert FAMILY.match(combat, None, {"check": "poison_ten"}) is None
