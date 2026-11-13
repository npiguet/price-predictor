"""The emittable-events table read against the connector's sources.

Each source set of ``emittable_events`` is the event types its Java shapers
name (``EventType.X``), so a shaper that starts or stops building a type fails
here rather than leaving a declaration reading every record of a kind as the
event's absence, as V5 #14 found.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.event_schema import EventType
from effects.domain.records import CollectionMode, Moment, RecordKind
from effects.domain.rule_detectors.emittable_events import (
    BUS,
    CLAUSE_HOOKS,
    RUN_PARAMS,
    absence_readable,
    carries,
)
from tests.unit.effects.builders import make_record

_SOURCES = (Path(__file__).resolve().parents[5] / "forge-connector" / "src" / "main" / "java"
            / "com" / "pricepredictor" / "connector" / "effects")


def _named(*files: str) -> set[EventType]:
    found: set[EventType] = set()
    for name in files:
        path = _SOURCES / f"{name}.java"
        if not path.exists():
            pytest.skip(f"no connector source at {path}")
        source = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
        found |= {EventType[m] for m in re.findall(r"EventType\.([A-Z_]+)\b", source)}
    return found


def _outcome_hook_types() -> set[EventType]:
    path = _SOURCES / "ClauseOutcomeRouter.java"
    if not path.exists():
        pytest.skip(f"no connector source at {path}")
    source = path.read_text(encoding="utf-8")
    body = source.split("KNOWN_OUTCOME_TYPES = EnumSet.of(", 1)[1].split(");", 1)[0]
    return {EventType[m] for m in re.findall(r"EventType\.([A-Z_]+)", body)}


def test_the_bus_set_is_what_the_bus_shapers_build():
    assert BUS == _named("BusEvents", "BusEventReader", "StatDiffer")


def test_the_clause_hook_set_is_what_the_clause_hooks_build():
    hooks = _named("ApiEvents") | _outcome_hook_types() | {EventType.MANA_PRODUCED}
    assert CLAUSE_HOOKS == hooks


def test_the_run_parameter_set_is_what_the_normalizer_builds():
    assert RUN_PARAMS == _named("RunParamNormalizer")


def test_the_game_s_end_reads_on_a_game_ending_line_alone():
    assert carries("resolution", "player_lost", acting_line=True, api="LosesGame")
    assert not carries("resolution", "player_lost", acting_line=True, api="Draw")
    assert carries("resolution", "player_lost")


def test_a_protection_fork_s_records_read_absences_as_any_record_does():
    """A forced resolution's lone half heard no clause hook; a protection
    fork's linked halves were played forward with the hooks routed to them."""
    forced = make_record(moment=Moment.RESOLUTION, interventional=True, fork=True,
                         mode=CollectionMode.PATCHED)
    protection = make_record(moment=Moment.RESOLUTION, interventional=True, fork=True,
                             mode=CollectionMode.PATCHED, link_id="L1")
    assert not absence_readable(forced, EventType.TOKEN_CREATED)
    assert absence_readable(protection, EventType.TOKEN_CREATED)
    assert absence_readable(forced, EventType.MANA_PRODUCED)
    trigger = make_record(RecordKind.TRIGGER, interventional=True, fork=True,
                          mode=CollectionMode.PATCHED)
    assert absence_readable(trigger, EventType.TOKEN_CREATED)
