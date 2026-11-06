"""Detector family ``trigger_mode``: a trigger evaluated, fired or not.

A ``trigger`` record is written whenever Forge evaluates a trigger against an
event, so the record itself says which side of a triggering rule it is on:
``fired`` true is the rule's ``fires`` side, false its near miss. The trigger
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
- ``origin``, ``destination``: a ``ChangesZone`` trigger's zones;
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
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    acting_root_params,
    is_keyword_line,
    script_params_match,
    source_entities,
    source_ids,
)
from effects.domain.rule_detectors.entity_facts import has_designation, is_copy
from effects.domain.rule_outcomes import PHASE_ORDER
from effects.domain.script_text import api_of, params, segments

__all__ = ["FAMILY", "acting_root_params"]

_SCRIPT_PARAMS = {"mode": "Mode", "origin": "Origin", "destination": "Destination",
                  "selector": "ValidCard", "phase": "Phase"}
#: The parameters read off the trigger's script segment.
_SEGMENT_PARAMS = frozenset(_SCRIPT_PARAMS) | {"script_param"}
#: The chain APIs that create a trigger while resolving.
SPAWNING_APIS = frozenset({"DelayedTrigger", "ImmediateTrigger"})


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
class TriggerMode:
    id: str = "trigger_mode"
    params: frozenset[str] = _SEGMENT_PARAMS | {
        "keyword", "spawned_by", "source_is_copy", "source_designation", "step"}
    general: frozenset[str] = frozenset({"mode", "keyword", "spawned_by", "step"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset({"designations"}) if "source_designation" in params else frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        keyword = params.get("keyword")
        spawned_by, step = params.get("spawned_by"), params.get("step")
        if keyword is not None and set(params) & _SEGMENT_PARAMS:
            raise ValueError("trigger_mode: a keyword's line has no script parameters to compare")
        if keyword is not None and spawned_by is not None:
            raise ValueError("trigger_mode: a keyword's line holds no chain to spawn a trigger")
        if keyword is None and "mode" not in params and step is None and spawned_by is None:
            raise ValueError(
                "trigger_mode: a declaration names the trigger's mode, keyword, step or spawned_by")
        if spawned_by is not None and spawned_by not in SPAWNING_APIS:
            raise ValueError(f"trigger_mode: spawned_by is one of {sorted(SPAWNING_APIS)}")
        if step is not None and step not in PHASE_ORDER:
            raise ValueError(f"trigger_mode: step {step!r} is none of {PHASE_ORDER}")
        if "named" in str(params.get("selector", "")).lower():
            raise ValueError("trigger_mode: a selector naming a card fixes one scenario's board")
        if record.kind is not RecordKind.TRIGGER:
            return None
        if step is not None and (record.state.global_.phase or "").lower() != step:
            return None
        if not self._source_fits(record, params):
            return None
        if keyword is not None:
            if not is_keyword_line(acting_line(record, lines), keyword):
                return None
            return self._side(record, params)
        if spawned_by is not None:
            found = _spawning_segment(record, lines, spawned_by)
        elif set(params) & _SEGMENT_PARAMS:
            found = acting_root_params(record, lines)
        else:
            found = {}
        if found is None:
            return None
        for name, script_name in _SCRIPT_PARAMS.items():
            if name in params and found.get(script_name) != params[name]:
                return None
        if "script_param" in params and not script_params_match(found, params["script_param"]):
            return None
        return self._side(record, params)

    @staticmethod
    def _source_fits(record: EffectRecord, params: Mapping[str, Any]) -> bool:
        """Whether the trigger's source is a copy where ``source_is_copy``
        asks; a record with no readable source fits no such declaration."""
        if "source_is_copy" not in params:
            return True
        sources = source_entities(record.state, record.ability or ())
        return bool(sources) and any(
            is_copy(entity, record.state) == bool(params["source_is_copy"])
            for entity in sources)

    @staticmethod
    def _side(record: EffectRecord, params: Mapping[str, Any]) -> FamilyMatch | None:
        fired = record.payload.fired
        designation = params.get("source_designation")
        if designation is not None:
            held = [has_designation(entity, designation)
                    for entity in source_entities(record.state, record.ability or ())]
            held = [h for h in held if h is not None]
            if not held or any(held) != fired:
                return None
        return FamilyMatch(Polarity.FIRES if fired else Polarity.DOES_NOT_FIRE, fired,
                           tuple(dict.fromkeys((*record.payload.event.subjects,
                                                *source_ids(record)))))


FAMILY = register(TriggerMode())
