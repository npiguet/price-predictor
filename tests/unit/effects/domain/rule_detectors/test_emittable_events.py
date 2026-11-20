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
    BUFFERED,
    BUS,
    CHOKE_POINTS,
    CLAUSE_HOOKS,
    EMITTABLE,
    RUN_PARAMS,
    TAPPED,
    absence_readable,
    carries,
    carries_buffered,
)
from tests.unit.effects.builders import java_golden_records, make_record

_SOURCES = (Path(__file__).resolve().parents[5] / "forge-connector" / "src" / "main" / "java"
            / "com" / "pricepredictor" / "connector" / "effects")

#: The classes holding ``ApiEvents``' rules, one family of APIs each.
_API_RULE_FAMILIES = (
    "TurnAndPhaseApiRules", "ValueAndOwnershipApiRules", "ChoiceApiRules",
    "HealGrantAndRetargetApiRules", "ScheduleAndReplacementApiRules", "TitleApiRules",
    "CombatRemovalApiRules", "TextApiRules", "CombatEntryAndSearchApiRules",
    "PermanentStateApiRules",
)


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
    assert BUS == _named("BusEvents", "BusEventReader", "StatDiffer", "ControlCorrection")


def test_the_clause_hook_set_is_what_the_clause_hooks_build():
    hooks = _named("ApiEvents", *_API_RULE_FAMILIES) | _outcome_hook_types() | {EventType.MANA_PRODUCED}
    assert CLAUSE_HOOKS == hooks


def test_the_run_parameter_set_is_what_the_normalizer_builds():
    assert RUN_PARAMS == _named("RunParamNormalizer", "RunParamRows")


def test_the_game_s_end_reads_on_a_game_ending_line_alone():
    assert carries("resolution", "player_lost", acting_line=True, api="LosesGame")
    assert not carries("resolution", "player_lost", acting_line=True, api="Draw")
    assert carries("resolution", "player_lost")


def test_every_fork_s_records_read_absences_as_any_patched_record_does():
    """A forced resolution's lone half hears the clause and outcome hooks
    (``ForkHooks``), as a protection fork's linked halves, played forward with
    the hooks routed to them, do: a missing hook-derived event did not
    happen. On a degraded worker's record it says nothing."""
    forced = make_record(moment=Moment.RESOLUTION, interventional=True, fork=True,
                         mode=CollectionMode.PATCHED)
    protection = make_record(moment=Moment.RESOLUTION, interventional=True, fork=True,
                             mode=CollectionMode.PATCHED, link_id="L1")
    for event in (EventType.TOKEN_CREATED, EventType.DAMAGE_PREVENTED, EventType.MANA_PRODUCED):
        assert absence_readable(forced, event)
        assert absence_readable(protection, event)
    trigger = make_record(RecordKind.TRIGGER, interventional=True, fork=True,
                          mode=CollectionMode.PATCHED)
    assert absence_readable(trigger, EventType.TOKEN_CREATED)
    degraded = make_record(moment=Moment.RESOLUTION, interventional=True, fork=True,
                           mode=CollectionMode.DEGRADED)
    assert not absence_readable(degraded, EventType.TOKEN_CREATED)


def _source(name: str) -> str:
    path = _SOURCES / f"{name}.java"
    if not path.exists():
        pytest.skip(f"no connector source at {path}")
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)


def test_the_bracket_s_taps_are_the_mana_and_rewrite_channels_and_the_fork_premise():
    """What a tap takes is what the bracket routes: the bus's events (its
    damage, moves and outcomes) and the choke points', asked of the taps first.
    The fork premise's record is a turn action, whose kind lists all of them."""
    taps = {path.stem for path in _SOURCES.glob("*.java")
            if re.search(r"implements\s+EventRouter\.EventTap", _source(path.stem))}
    assert taps == {"ManaChannel", "RewriteChannel", "ForkPremise"}
    assert TAPPED <= EMITTABLE[RecordKind.TURN_ACTION]
    route = _source("EventRouter").split("void route(", 1)[1]
    assert route.index("tap.take(") < route.index("classifyNow()")
    assert TAPPED == BUS | CHOKE_POINTS


def test_a_mana_record_carries_its_payment_s_and_sub_abilities_events():
    """A sacrificed creature's move, a pain land's damage: a mana record is a
    resolution, whose kind lists every event the mana channel builds or takes."""
    assert _named("ManaChannel") | TAPPED <= EMITTABLE[RecordKind.RESOLUTION]
    for event in ("tapped", "sacrificed", "zone_change", "mana_produced",
                  "damage_dealt", "life_change"):
        assert carries("resolution", event)
    mana = [raw for raw in java_golden_records() if raw.get("moment") == "resolution"
            and any(e["type"] == "mana_produced" for e in raw["payload"]["events"])]
    assert mana and all(carries("resolution", e["type"])
                        for raw in mana for e in raw["payload"]["events"])


def test_a_set_aside_ability_s_run_is_a_place_of_its_own():
    """``buffered_events`` hold the run the rewrite channel tapped, never what
    a rewrite record's own run parameters build."""
    assert BUFFERED == TAPPED
    assert carries_buffered("card_drawn") and not carries_buffered("token_created")
    runs = [e for raw in java_golden_records() if raw["kind"] == "rewrite"
            for e in raw["payload"].get("buffered_events", ())]
    assert runs and all(carries_buffered(e["type"]) for e in runs)


def test_a_clause_a_replacement_runs_in_a_window_files_its_events_on_that_window_s_record():
    """With no resolution bracket open, ``EventRouter.route`` files a
    clause's events (``Filing.CLAUSE``) on the open combat damage step's
    record, else in the turn-based action's window: a regeneration shield
    used by lethal combat damage writes its clause events on the combat record
    (``CombatReplacementClauseTest``)."""
    route = _source("EventRouter").split("private void fileConsequence(", 1)[1]
    assert route.index("combatStep.file(") < route.index("turnActions.take(")
    assert "filing != Filing.NON_COMBAT_DAMAGE" in route
    for kind in ("combat", "turn_action"):
        for event in CLAUSE_HOOKS:
            assert carries(kind, event), (kind, event)
    assert EMITTABLE[RecordKind.COMBAT] == EMITTABLE[RecordKind.TURN_ACTION]
    for event in ("regenerated", "removed_from_combat", "token_created", "replacement_applied"):
        assert carries("combat", event)
