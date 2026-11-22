"""Life loss, the game-end record and the game-end rules on combat records."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import EventType
from effects.domain.rule_detectors import sba_game_loss
from effects.domain.rule_detectors.outcome_event import FAMILY as OUTCOME

_GAME_END = (EventType.PLAYER_WON, EventType.PLAYER_LOST)


def _player_hit(by_kind):
    """A real combat damaging a player, and that player."""
    for record in by_kind("combat"):
        hit = [e.subjects[0] for e in record.payload.events
               if e.type is EventType.DAMAGE_DEALT and e.subjects[0].startswith("P")]
        if hit and not record.fork:
            return record, hit[0]
    pytest.fail("no combat damages a player")


def _infect_hit(by_kind):
    """That combat edited as infect deals it: the damage event stays, the
    player's life_change is gone, and the player stands at 20 life."""
    record, player = _player_hit(by_kind)
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.LIFE_CHANGE and player in e.subjects))
    state = replace(record.state, players=tuple(
        replace(p, life=20) if p.id == player else p for p in record.state.players))
    return replace(record, state=state, payload=replace(record.payload, events=events)), player


def test_infect_damage_to_a_player_at_20_life_tags_nothing_for_704_5a(by_kind):
    record, _ = _infect_hit(by_kind)
    assert sba_game_loss.FAMILY.match(record, None, {"check": "life_zero"}) is None


def test_damage_with_no_life_change_is_no_life_loss_even_at_1_life(by_kind):
    record, player = _infect_hit(by_kind)
    state = replace(record.state, players=tuple(
        replace(p, life=1) if p.id == player else p for p in record.state.players))
    assert sba_game_loss.FAMILY.match(replace(record, state=state), None,
                                      {"check": "life_zero"}) is None


def _game_end_record(gen3_records):
    return next(r for r in gen3_records if r.is_game_end_record)


def _with_event(record, event_type, **params):
    from effects.domain.event_schema import Event
    event = Event(type=event_type, subjects=("P0",), params=params)
    return replace(record, payload=replace(record.payload,
                                           events=(*record.payload.events, event)))


def test_a_game_end_record_tags_the_game_end_and_nothing_else(gen3_records):
    """A real game-end record, edited to hold a life change as well (the
    collector rescues events no record held into it)."""
    record = _with_event(_game_end_record(gen3_records), EventType.LIFE_CHANGE, delta=1)
    assert OUTCOME.match(record, None, {"event": "player_won"}) is not None
    assert OUTCOME.match(record, None, {"event": "player_lost"}) is not None
    assert OUTCOME.match(record, None, {"event": "life_change"}) is None
    assert OUTCOME.match(record, None, {"event": "zone_change"}) is None


def test_the_same_event_on_an_ordinary_effect_half_still_tags(gen3_records):
    record = _with_event(_game_end_record(gen3_records), EventType.LIFE_CHANGE, delta=1)
    ordinary = replace(record, link_id="link")
    assert not ordinary.is_game_end_record
    assert OUTCOME.match(ordinary, None, {"event": "life_change"}) is not None
