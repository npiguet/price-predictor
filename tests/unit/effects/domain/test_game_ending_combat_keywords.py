"""Gate 2's table reads no combat that ends the game (decision D13, finding 9.7).

Forge's state-based check tests the game-over condition first and returns
before any creature is destroyed, so a combat whose damage ends the game shows
no creature death (704.3 performs every state-based action at once; Forge does
not). Gate 2's observations, the survey's keyword threshold and the
``keyword_combat`` detector all read ``qualifying_observations``, which leaves
such a record out by the test ``sba_zone_move`` uses (``sba_board.game_ends``).

The records are the fixture's real game-ending combats. None of them holds a
carrier the table qualifies, so the one that does is a real one with a single
keyword granted (no real case is in the fixture).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.board_edits import BoardEdit
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KeywordResolver,
    qualifying_observations,
)
from effects.domain.event_schema import EventType
from effects.domain.records import RecordKind
from effects.domain.rule_detectors.sba_board import game_ends
from effects.infrastructure.shard_io import read_shard

_FIXTURE = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"
#: A live combat whose damage takes P1 from 6 to -2 life: E56 (1/2) blocks E25 (4/6).
_ENDING = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv.425"
_ENDING_EVENTS = (EventType.LIFE_CHANGE, EventType.PLAYER_LOST, EventType.PLAYER_WON)


@pytest.fixture(scope="module")
def records():
    return {record.record_id: record for record in read_shard(_FIXTURE)}


def _deathtouch_blocker(records):
    """The real game-ending combat, its blocker E56 granted deathtouch: its 1
    damage to E25 would leave E25 alive normally, a deathtouch observation."""
    return BoardEdit(records[_ENDING]).grant_keyword("E56", "deathtouch").record


def _without_the_end(record):
    """The same combat with the events that end the game taken out."""
    kept = tuple(e for e in record.payload.events if e.type not in _ENDING_EVENTS)
    return replace(record, payload=replace(record.payload, events=kept))


def test_a_qualifying_carrier_in_a_game_ending_combat_is_no_observation(records):
    record = _deathtouch_blocker(records)
    assert game_ends(record)
    assert qualifying_observations(record, DAMAGE_STEP_KEYWORDS, KeywordResolver()) == []


def test_the_same_carrier_qualifies_once_the_game_goes_on(records):
    record = _without_the_end(_deathtouch_blocker(records))
    assert not game_ends(record)
    found = qualifying_observations(record, DAMAGE_STEP_KEYWORDS, KeywordResolver())
    assert [(row.keyword, p.carrier.id) for row, p in found] == [("deathtouch", "E56")]


def test_no_game_ending_combat_of_the_fixture_is_observed(records):
    ending = [r for r in records.values() if r.kind is RecordKind.COMBAT and game_ends(r)]
    assert ending
    for record in ending:
        assert qualifying_observations(record, DAMAGE_STEP_KEYWORDS, KeywordResolver()) == []
