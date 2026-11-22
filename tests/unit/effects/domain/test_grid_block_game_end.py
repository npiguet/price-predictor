"""The combat grids pair no creatures of a combat that ends the game (decision
D13, finding 9.7).

Forge destroys no creature in a combat whose damage ends the game, so the
outcome a grid reads there is not the rule's. ``grid_block`` leaves such a
record out by the test ``sba_zone_move`` uses (``sba_board.game_ends``),
before any pair is read.

The record is the fixture's real game-ending combat in which E218 (1/1) blocks
E65 (1/2). Each deals the other 1, so E218 dies by the rule and the record
shows no death: the pair fails the outcome check already. Its blocker is given
a toughness of 2 so that no death is due and only the game's end refuses it
(no real case is in the fixture).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.board_edits import BoardEdit
from effects.domain.event_schema import EventType
from effects.domain.grid_edits import grid_block
from effects.domain.rule_detectors.sba_board import game_ends
from effects.infrastructure.shard_io import read_shard

_FIXTURE = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"
_ENDING = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.627"
_ENDING_EVENTS = (EventType.LIFE_CHANGE, EventType.PLAYER_LOST, EventType.PLAYER_WON)


def _no_keywords(entity) -> frozenset[str]:
    return frozenset()


@pytest.fixture(scope="module")
def surviving_block():
    record = next(r for r in read_shard(_FIXTURE) if r.record_id == _ENDING)
    return BoardEdit(record).set_toughness("E218", 2).record


def test_a_block_in_a_game_ending_combat_is_no_grid_pair(surviving_block):
    assert game_ends(surviving_block)
    assert grid_block(surviving_block, _no_keywords, "one") is None


def test_the_same_block_pairs_once_the_game_goes_on(surviving_block):
    kept = tuple(e for e in surviving_block.payload.events if e.type not in _ENDING_EVENTS)
    going_on = replace(surviving_block, payload=replace(surviving_block.payload, events=kept))
    assert not game_ends(going_on)
    assert grid_block(going_on, _no_keywords, "one") == ("E65", "E218")
