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

from effects.domain.effect_head_input import DESIGNATIONS
from effects.domain.effect_targets import _STATE_FLAG_FIELDS
from effects.domain.records import ForkKind
from effects.domain.rule_detectors import declared_values
from effects.domain.state_snapshot import (
    DayNight,
    EffectDuration,
    EntityRole,
    FaceState,
    LineKind,
)
from effects.domain.wire_closed_lists import (
    ABSENT,
    CHOICE_KINDS,
    COPY_DROPPED_KINDS,
    DAMAGE_TYPES,
    EVENT_DURATIONS,
    FORGE_PHASES,
    FORGE_ZONES,
    GAME_LOSS_REASONS,
    off_list_values,
)
from tests.unit.effects.builders import EFFECTS_FIXTURES, java_golden_records
from tests.unit.forge_sources import FORGE_AI, FORGE_GAME, forge_source


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
    (_record(entities=[_entity(lines=[
        {"keys": [], "kind": "keyword", "granted": True, "grant": "anthem"}])]),
     ("line.grant", "anthem")),
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
    (_record(pending_event={"type": "face_change", "params": {"to_state": "transformed"}}),
     ("face_change.to_state", "transformed")),
    (_record(pending_event={"type": "damage_dealt", "params": {"damage_type": "infect"}}),
     ("damage_dealt.damage_type", "infect")),
    (_record(pending_event={"type": "player_lost", "params": {"reason": "Decked"}}),
     ("player_lost.reason", "Decked")),
    (_record(pending_event={"type": "choice_made", "params": {"choice_kind": "colour"}}),
     ("choice_made.choice_kind", "colour")),
    (_record(pending_event={"type": "pt_change", "duration": "until_leaves"}),
     ("event.duration", "until_leaves")),
    (_record() | {"fork": True}, ("fork_kind", ABSENT)),
    (_record() | {"fork": True, "fork_kind": "replay"}, ("fork_kind", "replay")),
])
def test_a_value_outside_its_list_is_named(raw, found):
    assert list(off_list_values(raw)) == [found]


def test_every_payload_s_events_are_read():
    raw = _record() | {"kind": "rewrite", "payload": {
        "incoming": {"type": "zone_change", "params": {"to_zone": "outside"}},
        "outgoing": {"type": "zone_change", "params": {"to_zone": "merged"}}}}
    assert list(off_list_values(raw)) == [("zone_change.to_zone", "outside")]


def test_a_buffered_ability_s_events_are_read():
    """A rewrite's ``buffered_events`` are the set-aside ability's run (WP7)."""
    raw = _record() | {"kind": "rewrite", "payload": {
        "incoming": {"type": "damage_dealt", "params": {"amount": 2}},
        "outgoing": None, "buffered": [], "buffered_ran": True,
        "buffered_events": [{"type": "zone_change", "params": {"to_zone": "outside"},
                             "attributed_to": "root"}]}}
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
_FORGE_GAME = FORGE_GAME


def _enum_constants(path: Path, enum: str) -> tuple[str, ...]:
    """The constants an enum declares, read off its source: everything
    between the enum's brace and the first ``;`` or ``}``, each name before
    its ``(``."""
    forge_source(path)
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    # The constants end at the first ";", or at the enum's own "}" when it
    # declares nothing after them.
    body = re.split(r"[;}]", source.split(f"enum {enum}", 1)[1].split("{", 1)[1], maxsplit=1)[0]
    body = re.sub(r"\([^()]*\)", "", body)
    return tuple(name.strip() for name in body.split(",") if name.strip())


def test_the_zones_are_forge_s_zone_types_lower_cased():
    """``CardSpelling.zoneName`` writes ``ZoneType.name()`` lower-cased, so
    the list is Forge's enum and drifts with it only through this test."""
    names = _enum_constants(_FORGE_GAME / "zone" / "ZoneType.java", "ZoneType")
    assert sorted(FORGE_ZONES) == sorted(name.lower() for name in names)


def test_the_copy_dropped_kinds_are_the_copier_s_lower_cased():
    """``Forks.dropped`` writes ``GameCopier.NoCounterpart.name()``
    lower-cased, so the list is Forge's enum and drifts with it only through
    this test."""
    copier = FORGE_AI / "simulation" / "GameCopier.java"
    names = _enum_constants(copier, "NoCounterpart")
    assert sorted(COPY_DROPPED_KINDS) == sorted(name.lower() for name in names)


def test_the_connector_spells_the_copy_dropped_kinds_lower_cased():
    source = (_CONNECTOR / "Forks.java").read_text(encoding="utf-8")
    body = source.split("static Map<String, Integer> dropped(", 1)[1].split("return dropped;", 1)[0]
    assert "toLowerCase(Locale.ROOT)" in body


def test_the_game_loss_reasons_are_forge_s():
    names = _enum_constants(_FORGE_GAME / "player" / "GameLossReason.java", "GameLossReason")
    assert sorted(GAME_LOSS_REASONS) == sorted(names)


#: The connector's effects package, which writes the event parameters.
_CONNECTOR = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main"
              / "java" / "com" / "pricepredictor" / "connector" / "effects")


def test_the_damage_types_are_the_connector_s():
    source = (_CONNECTOR / "BusEvents.java").read_text(encoding="utf-8")
    body = source.split("static String damageType", 1)[1].split("static ", 1)[0]
    assert sorted(DAMAGE_TYPES) == sorted(re.findall(r'case \w+ -> "([a-z0-9_]+)"', body))


def test_the_choice_kinds_are_the_connector_s():
    source = (_CONNECTOR / "ChoiceApiRules.java").read_text(encoding="utf-8")
    assert sorted(CHOICE_KINDS) == sorted(re.findall(r'new ChoiceReading\("([a-z_]+)"', source))


def test_the_event_durations_are_the_connector_s():
    source = (_CONNECTOR / "EventDuration.java").read_text(encoding="utf-8")
    assert sorted(EVENT_DURATIONS) == sorted(re.findall(r'[A-Z_]+\("([a-z_]+)"\)', source))


def test_the_fork_kinds_are_the_connector_s_in_its_order():
    """``ForkKind.java``'s wire names; ``WireVocabularyTest.forkKinds``
    spells the same list out on the Java side."""
    source = (_CONNECTOR / "ForkKind.java").read_text(encoding="utf-8")
    assert list(ForkKind) == re.findall(r'^\s*[A-Z]+\("([a-z]+)"\)', source, re.MULTILINE)


def test_the_state_flags_are_the_connector_s():
    """``StateFlag.java``'s wire names: ``class_level`` (levels gained, no
    field of its own) then the binary flags ``_STATE_FLAG_FIELDS`` maps."""
    source = (_CONNECTOR / "StateFlag.java").read_text(encoding="utf-8")
    declared = re.findall(r'^\s*[A-Z_]+\("([a-z_]+)"\)[,;]', source, re.MULTILINE)
    assert declared == ["class_level", *_STATE_FLAG_FIELDS]


def _wire_names(file: str) -> list[str]:
    """The wire names a connector enum declares, in its order: each constant's
    first string argument."""
    source = (_CONNECTOR / file).read_text(encoding="utf-8")
    return re.findall(r'^\s*[A-Z_]+\("([a-z_]+)"[,)]', source, re.MULTILINE)


@pytest.mark.parametrize(("file", "vocabulary"), [
    ("FaceState.java", FaceState),
    ("EntityRole.java", EntityRole),
    ("LineKind.java", LineKind),
    ("FloatingDuration.java", EffectDuration),
    ("DayNight.java", DayNight),
])
def test_the_snapshot_s_vocabularies_are_the_connector_s_in_its_order(file, vocabulary):
    """``WireVocabularyTest`` spells each list out on the Java side."""
    assert list(vocabulary) == _wire_names(file)


def test_the_designations_are_the_connector_s_in_its_order():
    """``Designation.java``'s wire names, which ``WireVocabularyTest.designations``
    spells out on the Java side."""
    assert list(DESIGNATIONS) == _wire_names("Designation.java")


def test_the_two_python_copies_of_the_designations_are_equal():
    """``effect_head_input.py`` and ``rule_detectors/declared_values.py`` each
    list them; this task pins the Java enum against the first."""
    assert set(DESIGNATIONS) == declared_values.DESIGNATIONS


def test_a_live_record_s_absent_fork_kind_is_on_the_list():
    assert list(off_list_values(_record() | {"fork": False})) == []


def test_the_line_grants_are_the_connector_s():
    from effects.domain.state_snapshot import LineGrant

    source = (_CONNECTOR / "LineGrant.java").read_text(encoding="utf-8")
    assert sorted(LineGrant) == sorted(
        re.findall(r'static final String [A-Z]+ = "([a-z]+)";', source))


def test_a_printed_line_carries_no_grant_and_is_on_the_list():
    raw = _record(entities=[_entity(lines=[{"keys": [], "kind": "spell", "granted": False}])])
    assert list(off_list_values(raw)) == []


def test_a_last_known_entity_with_no_zone_is_on_the_list():
    raw = _record(entities=[_entity(id="L5", zone=None, role="last_known")])
    assert list(off_list_values(raw)) == []


def test_the_phases_are_forge_s_phase_types_lower_cased():
    names = _enum_constants(_FORGE_GAME / "phase" / "PhaseType.java", "PhaseType")
    assert sorted(FORGE_PHASES) == sorted(name.lower() for name in names)


def test_a_record_without_a_mode_is_named():
    """Absent, the decoder reads it as ``degraded``; every writer sets it."""
    raw = _record()
    del raw["mode"]
    assert list(off_list_values(raw)) == [("mode", ABSENT)]


def test_a_list_where_a_string_belongs_is_named_not_raised_on():
    assert list(off_list_values(_record(entities=[_entity(zone=["hand"])]))) == [
        ("entity.zone", ["hand"])]
