"""Detector family ``trigger_mode``: a trigger evaluated, fired or not.

A ``trigger`` record says which side of a triggering rule it is on:
``fired`` true is the rule's ``fires`` side, a trigger Forge ran (gen-3: its
condition held and it went on the stack, ``onTriggerRun``; an evaluation that
held and never ran writes no record), false its near miss, an evaluation
whose condition did not hold. The trigger
line's script, read through the sidecar, says which trigger it was.

Parameters, each a condition of the rule rather than of a card:

- ``mode``: the trigger's ``Mode$`` (``ChangesZone``, ``Attacks``,
  ``DamageDone`` …), required unless ``keyword``, ``spawned_by`` or ``step`` is given;
- ``keyword``: the trigger is one a keyword generates (exalted, evolve,
  cumulative upkeep), read off the acting line, which for such a trigger is the
  keyword's own line (``Exalted``, ``Suspend:2:…``) and carries no ``Mode$``; the
  name is compared in ``keyword_names``' spelling, so ``Cumulative upkeep`` and
  ``cumulative_upkeep`` agree (``acting_line.is_keyword_line``, the read
  every family's ``keyword`` shares). The script parameters below cannot be read off a
  keyword line, so a declaration giving ``keyword`` gives none of them;
- ``origin``, ``destination``: a ``ChangesZone`` trigger's zones, compared
  without case and as sets: a script's comma list holds a declared zone, and
  its ``Any`` holds a zone of ordinary play (``ORDINARY_ZONES``) the script's
  other side is not alone;
- ``phase``: a ``Phase`` trigger's ``Phase$`` (``EndCombat``, ``Upkeep``),
  compared as written;
- ``script_param``: the trigger's segment holds this parameter, a name or a
  name -> value map as ``outcome_event`` reads it (``{"IsPresent":
  "Card.Self+IsSolved"}``: a trigger working only while its Case is solved);
- ``spawned_by``: ``DelayedTrigger`` or ``ImmediateTrigger``: the trigger is
  one a resolving ability created (a delayed "at the beginning of the next
  end step", a reflexive "when you do"). Forge keys such a trigger to the
  line that made it, a spell, activated ability or static line whose chain
  holds the creating segment, so it is found by a chain segment of that API
  on a line that is no trigger line, and the script parameters above are
  read off that segment rather than the root;
- ``selector``: the trigger's ``ValidCard$`` selector, compared as written
  (``Creature.Other+YouCtrl``); one that names a card (``Card.namedX``) is
  refused, since it would match one scenario's board and nothing else.

Situations of the trigger's source (``refs.source``, else the battlefield
entity printed by the trigger's script file), and of the moment it was
evaluated:

- ``source_is_copy``: the source is (true) or is not (false) a copy of
  another object (``entity_facts.is_copy``: an enters trigger a Clone took
  from what it copies);
- ``source_designation``: a gen-3 designation the trigger's condition reads
  on its source (``harnessed``, ``solved``): a fired trigger counts only
  with its source holding it, an unfired one only with its source lacking
  it, so each side is the one the designation explains;
- ``source_subtype``: the trigger's source has this subtype (``Class``,
  ``Case``), compared without case;
- ``outcome_event``'s subject qualifiers (``subject_face_state``,
  ``subject_is_token`` …): the trigger's event subject is such an object in
  the snapshot; a record whose subjects cannot be read speaks for nothing;
- ``step``: the snapshot's phase or step when the trigger was evaluated, in
  ``rule_outcomes.PHASE_ORDER``'s spelling (``cleanup``, ``combat_damage``);
  given alone, it speaks for every trigger evaluated then.

The condition entities are the triggering event's subjects and the
trigger's source; a ``trigger`` record carries no outcome (what the trigger
does arrives as its resolution's record), so its tags name no outcome
entity and the rule's per-entity outputs are read nowhere on it.

Prior art: ``effect_families`` keys a trigger record by the same root
``Mode$``, read the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import EventlessFamily, FamilyMatch, Matcher, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    acting_root_params,
    is_keyword_line,
    script_params_match,
    source_entities,
    source_ids,
)
from effects.domain.rule_detectors.entity_facts import has_designation, is_copy
from effects.domain.rule_detectors.outcome_subjects import (
    QUALIFIER_CHECKS,
    SUBJECT_QUALIFIER_NEEDS,
    SUBJECT_QUALIFIERS,
    subject_qualifiers,
    subjects_meet,
)
from effects.domain.rule_outcomes import PHASE_ORDER
from effects.domain.script_text import api_of, params, segments
from effects.domain.state_snapshot import EntityState

__all__ = ["FAMILY", "acting_root_params"]

_SCRIPT_PARAMS = {"mode": "Mode", "origin": "Origin", "destination": "Destination",
                  "selector": "ValidCard", "phase": "Phase"}
#: The zones a script's ``Any`` holds.
ORDINARY_ZONES = frozenset({"hand", "library", "graveyard", "battlefield", "exile", "command",
                            "stack"})
#: The declared parameters that name zones.
_ZONE_PARAMS = frozenset({"origin", "destination"})
#: The parameters read off the trigger's script segment.
_SEGMENT_PARAMS = frozenset(_SCRIPT_PARAMS) | {"script_param"}
#: The chain APIs that create a trigger while resolving.
SPAWNING_APIS = frozenset({"DelayedTrigger", "ImmediateTrigger"})


def _validate_selection(params: Mapping[str, Any]) -> None:
    """Which triggers a declaration names: a keyword's line, a spawned trigger,
    a step or a mode, each with the parameters that go with it."""
    if params.get("keyword") is not None:
        _validate_keyword_line(params)
    if not ({"keyword", "mode", "step", "spawned_by"} & params.keys()):
        raise ValueError(
            "trigger_mode: a declaration names the trigger's mode, keyword, step or spawned_by")
    _validate_spawn_and_step(params.get("spawned_by"), params.get("step"))


def _validate_keyword_line(params: Mapping[str, Any]) -> None:
    if set(params) & _SEGMENT_PARAMS:
        raise ValueError("trigger_mode: a keyword's line has no script parameters to compare")
    if params.get("spawned_by") is not None:
        raise ValueError("trigger_mode: a keyword's line holds no chain to spawn a trigger")


def _validate_spawn_and_step(spawned_by: str | None, step: str | None) -> None:
    if spawned_by is not None and spawned_by not in SPAWNING_APIS:
        raise ValueError(f"trigger_mode: spawned_by is one of {sorted(SPAWNING_APIS)}")
    if step is not None and step not in PHASE_ORDER:
        raise ValueError(f"trigger_mode: step {step!r} is none of {PHASE_ORDER}")


def _zones(value: str) -> frozenset[str]:
    """A script's zone or comma list of zones, compared without case."""
    return frozenset(zone.strip().lower() for zone in value.split(","))


def _zone_fits(found: Mapping[str, str], name: str, wanted: str) -> bool:
    """Whether the script's ``Origin$`` or ``Destination$`` holds the declared
    zones: as a set, or, for ``Any``, as :func:`_any_holds` reads it."""
    have = found.get(_SCRIPT_PARAMS[name])
    if have is None:
        return False
    zones, declared = _zones(have), _zones(wanted)
    return _any_holds(found, name, declared) if "any" in zones else declared <= zones


def _any_holds(found: Mapping[str, str], name: str, declared: frozenset[str]) -> bool:
    """``Any`` holds a zone of ordinary play (:data:`ORDINARY_ZONES`: a rare
    zone's trigger is a rule of its own) other than the one the script's other
    side is alone, since no card moves from a zone to itself."""
    other = found.get(_SCRIPT_PARAMS["destination" if name == "origin" else "origin"])
    return declared <= ORDINARY_ZONES and (other is None or _zones(other) != declared)


def _param_fits(found: Mapping[str, str], name: str, wanted: str) -> bool:
    if name in _ZONE_PARAMS:
        return _zone_fits(found, name, wanted)
    return found.get(_SCRIPT_PARAMS[name]) == wanted


def _script_fits(found: Mapping[str, str], wanted: tuple[tuple[str, str], ...]) -> bool:
    """Whether the trigger's script holds each ``wanted`` value: a zone as
    :func:`_zone_fits` reads it, any other as written."""
    return all(_param_fits(found, name, value) for name, value in wanted)


def _spawning_segment(record: EffectRecord, lines: SidecarLookup | None,
                      api: str) -> dict[str, str] | None:
    """The parameters of the acting line's chain segment of the API ``api``,
    which created the trigger, or None where the record acts through a
    trigger line of its own or its line creates none."""
    keys = record.ability or ()
    if not keys or keys[0].trait_kind == "trigger":
        return None
    line = acting_line(record, lines)
    if line is None or not line.script_text:
        return None
    for segment in segments(line.script_text):
        found = params(segment)
        if api_of(found) == api:
            return found
    return None


@dataclass(frozen=True, slots=True)
class TriggerMode(EventlessFamily):
    id: str = "trigger_mode"
    kinds: frozenset[str] = frozenset({"trigger"})
    params: frozenset[str] = _SEGMENT_PARAMS | {
        "keyword", "spawned_by", "source_is_copy", "source_designation", "step",
        "source_subtype"} | frozenset(SUBJECT_QUALIFIERS)
    general: frozenset[str] = frozenset({"mode", "keyword", "spawned_by", "step"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        found = {field for name, field in SUBJECT_QUALIFIER_NEEDS.items() if name in params}
        if "source_designation" in params:
            found.add("designations")
        return frozenset(found)

    def validate(self, params: Mapping[str, Any]) -> None:
        _validate_selection(params)
        if "named" in str(params.get("selector", "")).lower():
            raise ValueError("trigger_mode: a selector naming a card fixes one scenario's board")
        dv.check(self.id, params, {
            **QUALIFIER_CHECKS, "mode": dv.trigger_mode, "origin": dv.zone,
            "destination": dv.zone, "phase": dv.forge_phase, "keyword": dv.keyword,
            "source_designation": dv.designation, "source_subtype": dv.subtype,
            "source_is_copy": dv.boolean, "script_param": dv.script_param})

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return TriggerMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


@dataclass(frozen=True, slots=True, eq=False)
class TriggerMatcher:
    """A compiled ``trigger_mode`` declaration."""

    keyword: str | None
    step: str | None
    #: ``source_subtype`` lower-cased, ``source_is_copy`` and
    #: ``source_designation``: None where absent.
    subtype: str | None
    is_copy: bool | None
    designation: str | None
    #: The subject qualifiers, name -> declared value.
    subjects: Mapping[str, Any]
    #: ``spawned_by``, None where absent.
    spawned_by: str | None
    #: Whether any parameter is read off the trigger's script segment.
    reads_segment: bool
    #: The script parameters declared, name -> value, in ``_SCRIPT_PARAMS`` order.
    script: tuple[tuple[str, str], ...]
    #: ``script_param`` where declared.
    script_param: Any

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> TriggerMatcher:
        subtype = params.get("source_subtype")
        return cls(
            params.get("keyword"), params.get("step"),
            subtype.lower() if subtype is not None else None,
            bool(params["source_is_copy"]) if "source_is_copy" in params else None,
            params.get("source_designation"), subject_qualifiers(params),
            params.get("spawned_by"), bool(set(params) & _SEGMENT_PARAMS),
            tuple((name, params[name]) for name in _SCRIPT_PARAMS if name in params),
            params.get("script_param"))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if record.kind is not RecordKind.TRIGGER or not self._selected(record, lines):
            return None
        return self._side(record)

    def _selected(self, record: EffectRecord, lines: SidecarLookup | None) -> bool:
        """Whether the trigger is one the declaration names, before its fired bit."""
        return (self._step_fits(record) and self._source_fits(record)
                and self._line_fits(record, lines))

    def _step_fits(self, record: EffectRecord) -> bool:
        if self.step is None:
            return True
        return (record.state.global_.phase or "").lower() == self.step

    def _line_fits(self, record: EffectRecord, lines: SidecarLookup | None) -> bool:
        """Whether the acting line is the ``keyword``'s, or its script holds
        the declared parameters."""
        if self.keyword is not None:
            return is_keyword_line(acting_line(record, lines), self.keyword)
        found = self._script_params(record, lines)
        if found is None or not _script_fits(found, self.script):
            return False
        return self.script_param is None or script_params_match(found, self.script_param)

    def _script_params(self, record: EffectRecord,
                       lines: SidecarLookup | None) -> dict[str, str] | None:
        """The script parameters a declaration's script conditions are read off:
        the spawning segment's under ``spawned_by``, else the acting line's root
        when any is declared, else none (an empty set)."""
        if self.spawned_by is not None:
            return _spawning_segment(record, lines, self.spawned_by)
        return acting_root_params(record, lines) if self.reads_segment else {}

    def _source_fits(self, record: EffectRecord) -> bool:
        """Whether the trigger's source is a copy where ``source_is_copy``
        asks; a record with no readable source fits no such declaration."""
        sources = source_entities(record.state, record.ability or ())
        if not self._subtype_fits(sources):
            return False
        if self.subjects and not subjects_meet(
                record.payload.event.subjects, record.state, self.subjects):
            # Not the object the rule is about, or none readable.
            return False
        return self.is_copy is None or any(
            is_copy(entity, record.state) == self.is_copy for entity in sources)

    def _subtype_fits(self, sources: tuple[EntityState, ...]) -> bool:
        wanted = self.subtype
        return wanted is None or any(
            wanted in {s.lower() for s in e.subtypes} for e in sources)

    def _side(self, record: EffectRecord) -> FamilyMatch | None:
        fired = record.payload.fired
        if self.designation is not None:
            held = [has_designation(entity, self.designation)
                    for entity in source_entities(record.state, record.ability or ())]
            held = [h for h in held if h is not None]
            if not held or any(held) != fired:
                return None
        return FamilyMatch(Polarity.FIRES if fired else Polarity.DOES_NOT_FIRE, fired,
                           tuple(dict.fromkeys((*record.payload.event.subjects,
                                                *source_ids(record)))))


FAMILY = register(TriggerMode())
