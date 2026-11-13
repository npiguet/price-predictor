"""The wire's closed lists, checked as a shard is read (TP2: closed lists at decode).

Each value is checked against the list the collector writes from, and the
real Java goldens and fixtures name none outside it, so a value outside is a
writer the decoder would otherwise read into a silent default.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from effects.domain.wire_closed_lists import (
    ABSENT,
    FORGE_PHASES,
    FORGE_ZONES,
    off_list_values,
)
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
    (_record(entities=[_entity(zone="sideboards")]), ("entity.zone", "sideboards")),
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
    (_record(pending_event={"type": "zone_change", "params": {"to_zone": "outside"}}),
     ("zone_change.to_zone", "outside")),
    (_record(pending_event={"type": "life_change", "attributed_to": "sub"}),
     ("event.attributed_to", "sub")),
])
def test_a_value_outside_its_list_is_named(raw, found):
    assert list(off_list_values(raw)) == [found]


def test_every_payload_s_events_are_read():
    raw = _record() | {"kind": "rewrite", "payload": {
        "incoming": {"type": "zone_change", "params": {"to_zone": "outside"}},
        "outgoing": {"type": "zone_change", "params": {"to_zone": "merged"}}}}
    assert list(off_list_values(raw)) == [("zone_change.to_zone", "outside")]


@pytest.mark.parametrize("zone", [*FORGE_ZONES, None])
def test_every_zone_forge_names_is_on_the_list_for_a_move_and_an_entity(zone):
    """``merged``, ``sideboard`` and the rest are zones the collector really
    writes; null is a last-known card no zone holds, or an unnamed end."""
    raw = _record(entities=[_entity(zone=zone)]) | {"payload": {"events": [
        {"type": "zone_change", "params": {"from_zone": zone, "to_zone": zone}}]}}
    assert list(off_list_values(raw)) == []


@pytest.mark.parametrize("phase", [*FORGE_PHASES, None])
def test_every_phase_forge_names_is_on_the_list(phase):
    assert list(off_list_values(_record(**{"global": {"phase": phase}}))) == []


#: The sibling Forge checkout the collector compiles against.
_FORGE_GAME = (Path(__file__).resolve().parents[4].parent / "forge" / "forge-game" / "src"
               / "main" / "java" / "forge" / "game")


def _enum_constants(path: Path, enum: str) -> tuple[str, ...]:
    """The constants an enum declares, read off its source: everything
    between the enum's brace and the first ``;``, each name before its ``(``."""
    if not path.exists():
        pytest.skip(f"no Forge checkout at {path}")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    body = source.split(f"enum {enum}", 1)[1].split("{", 1)[1].split(";", 1)[0]
    body = re.sub(r"\([^()]*\)", "", body)
    return tuple(name.strip() for name in body.split(",") if name.strip())


def test_the_zones_are_forge_s_zone_types_lower_cased():
    """``CardSpelling.zoneName`` writes ``ZoneType.name()`` lower-cased, so
    the list is Forge's enum and drifts with it only through this test."""
    names = _enum_constants(_FORGE_GAME / "zone" / "ZoneType.java", "ZoneType")
    assert sorted(FORGE_ZONES) == sorted(name.lower() for name in names)


def test_the_phases_are_forge_s_phase_types_lower_cased():
    names = _enum_constants(_FORGE_GAME / "phase" / "PhaseType.java", "PhaseType")
    assert sorted(FORGE_PHASES) == sorted(name.lower() for name in names)


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
