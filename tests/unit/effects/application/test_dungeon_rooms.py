"""A dungeon's rooms are ability slots, the current room flagged active.

A dungeon is a command-zone card whose one keyword line reads as one slot per
room line (``board_keys``), each room a ``RoomEntered`` trigger of the
dungeon's token script naming its room in ``ValidRoom$``. The snapshot names
the room the player is in (``global.dungeons[].room``); the room slot of
that name is flagged active and every other room slot inactive, so the
dungeon reads which room comes next.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.input_layout import SlotKind
from tests.unit.effects.application.test_surface_batching_option_rows import _fixture_batcher

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: Tomb of Annihilation, E81: P0 is in its first room, Trapped Entry.
_IN_THE_TOMB = f"{_RUN}.2-mwqvwtly1zvq.1597"
_ROOMS = ("Trapped Entry", "Veils of Fear", "Oubliette", "Sandfall Cell",
          "Cradle of the Death God")


@pytest.fixture(scope="module")
def batcher_and_record():
    batcher, records = _fixture_batcher()
    return batcher, next(record for record in records if record.record_id == _IN_THE_TOMB)


def _in_room(record, room: str):
    g = record.state.global_
    dungeons = tuple(replace(dungeon, room=room) for dungeon in g.dungeons)
    return replace(record, state=replace(record.state, global_=replace(g, dungeons=dungeons)))


def _room_slots(batcher, record):
    texts = batcher.batch_texts([record])
    rows = {text: row for row, text in enumerate(sorted(texts))}
    surface = batcher.surface_builder(rows).build(record)
    return [slot for slot in surface.slots
            if slot.kind is SlotKind.ABILITY and slot.entity_id == "E81"]


def _active(slots) -> list[float]:
    return [slot.features[0] for slot in slots]


def test_each_room_line_names_its_room(batcher_and_record):
    batcher, record = batcher_and_record
    tomb = record.state.entity("E81")
    keys = batcher.board_keys(tomb.lines[0].keys[0])
    assert tuple(batcher.room_of(key) for key in keys) == _ROOMS


def test_the_current_room_is_the_one_active_room(batcher_and_record):
    batcher, record = batcher_and_record
    assert [d.room for d in record.state.global_.dungeons] == ["Trapped Entry"]
    assert _active(_room_slots(batcher, record)) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_another_room_reads_apart(batcher_and_record):
    batcher, record = batcher_and_record
    deeper = _in_room(record, "Veils of Fear")
    assert _active(_room_slots(batcher, deeper)) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert _room_slots(batcher, record) != _room_slots(batcher, deeper)
