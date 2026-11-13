"""The wire's closed lists, checked as a shard is read (TP2: closed lists at decode).

Each value is checked against the list the collector writes from, and the
real Java goldens and fixtures name none outside it, so a value outside is a
writer the decoder would otherwise read into a silent default.
"""

from __future__ import annotations

import gzip
import json

import pytest

from effects.domain.wire_closed_lists import ABSENT, off_list_values
from tests.unit.effects.builders import EFFECTS_FIXTURES, java_golden_records


def _fixture(name: str) -> list[dict]:
    with gzip.open(EFFECTS_FIXTURES / f"{name}-records.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_java_goldens_name_nothing_outside_the_lists():
    assert [found for raw in java_golden_records() for found in off_list_values(raw)] == []


def test_the_gen3_fixture_names_nothing_outside_the_lists():
    """``merged`` is a mutated permanent's merged card's zone, a value of the list."""
    assert [found for raw in _fixture("gen3") for found in off_list_values(raw)] == []


def _record(**state) -> dict:
    return {
        "record_id": "r", "generation": 3, "mode": "patched", "kind": "resolution",
        "state": {"global": {"phase": "main1"}, "entities": [], **state},
        "payload": {"events": []},
    }


def _entity(**members) -> dict:
    return {"id": "E1", "name": "x", "zone": "battlefield", **members}


@pytest.mark.parametrize(("raw", "found"), [
    (_record(entities=[_entity(zone="sideboard")]), ("entity.zone", "sideboard")),
    (_record(entities=[_entity(face_state="transformed")]), ("entity.face_state", "transformed")),
    (_record(entities=[_entity(role="bystander")]), ("entity.role", "bystander")),
    (_record(entities=[_entity(lines=[{"keys": [], "kind": "ability"}])]),
     ("line.kind", "ability")),
    (_record(links=[{"from": "E1", "to": "E2", "type": "haunts"}]), ("link.type", "haunts")),
    (_record(**{"global": {"phase": "beginning"}}), ("global.phase", "beginning")),
    (_record(**{"global": {"phase": "main2", "combat_substep": "trample"}}),
     ("global.combat_substep", "trample")),
    (_record(**{"global": {"phase": "main2", "day_night": "dusk"}}), ("global.day_night", "dusk")),
    (_record(**{"global": {"phase": "main2", "floating_effects": [{"duration": "forever"}]}}),
     ("global.floating_effects.duration", "forever")),
    (_record(pending_event={"type": "zone_change", "params": {"to_zone": "sideboard"}}),
     ("zone_change.to_zone", "sideboard")),
    (_record(pending_event={"type": "life_change", "attributed_to": "sub"}),
     ("event.attributed_to", "sub")),
])
def test_a_value_outside_its_list_is_named(raw, found):
    assert list(off_list_values(raw)) == [found]


def test_every_payload_s_events_are_read():
    raw = _record() | {"kind": "rewrite", "payload": {
        "incoming": {"type": "zone_change", "params": {"to_zone": "graveyard"}},
        "outgoing": {"type": "zone_change", "params": {"to_zone": "merged"}}}}
    assert list(off_list_values(raw)) == [("zone_change.to_zone", "merged")]


def test_a_move_to_the_stack_or_to_no_zone_is_on_the_list():
    raw = _record() | {"payload": {"events": [
        {"type": "zone_change", "params": {"from_zone": "hand", "to_zone": "stack"}},
        {"type": "zone_change", "params": {"from_zone": None, "to_zone": "none"}}]}}
    assert list(off_list_values(raw)) == []


def test_a_gen3_record_without_a_mode_is_named():
    """Absent, the decoder reads it as ``degraded``; every gen-3 writer sets it."""
    raw = _record()
    del raw["mode"]
    assert list(off_list_values(raw)) == [("mode", ABSENT)]
    del raw["generation"]
    assert list(off_list_values(raw)) == []


def test_a_list_where_a_string_belongs_is_named_not_raised_on():
    assert list(off_list_values(_record(entities=[_entity(zone=["hand"])]))) == [
        ("entity.zone", ["hand"])]
