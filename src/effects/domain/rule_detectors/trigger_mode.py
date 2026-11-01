"""Detector family ``trigger_mode``: a trigger evaluated, fired or not.

A ``trigger`` record is written whenever Forge evaluates a trigger against an
event, so the record itself says which side of a triggering rule it is on:
``fired`` true is the rule's ``fires`` side, false its near miss. The trigger
line's script, read through the sidecar, says which trigger it was.

Parameters, each a condition of the rule rather than of a card:

- ``mode``: the trigger's ``Mode$`` (``ChangesZone``, ``Attacks``,
  ``DamageDone`` …), required unless ``keyword`` is given;
- ``keyword``: the trigger is one a keyword generates (exalted, evolve,
  cumulative upkeep), read off the acting line, which for such a trigger is the
  keyword's own line (``Exalted``, ``Suspend:2:…``) and carries no ``Mode$``; the
  name is compared in ``keyword_names``' spelling, so ``Cumulative upkeep`` and
  ``cumulative_upkeep`` agree (``acting_line.is_keyword_line``, the read
  every family's ``keyword`` shares). The script parameters below cannot be read off a
  keyword line, so a declaration giving ``keyword`` gives none of them;
- ``origin``, ``destination``: a ``ChangesZone`` trigger's zones;
- ``selector``: the trigger's ``ValidCard$`` selector, compared as written
  (``Creature.Other+YouCtrl``); one that names a card (``Card.namedX``) is
  refused, since it would match one scenario's board and nothing else.

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
)

__all__ = ["FAMILY", "acting_root_params"]

_SCRIPT_PARAMS = {"mode": "Mode", "origin": "Origin", "destination": "Destination",
                  "selector": "ValidCard"}


@dataclass(frozen=True, slots=True)
class TriggerMode:
    id: str = "trigger_mode"
    params: frozenset[str] = frozenset(_SCRIPT_PARAMS) | {"keyword"}

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        keyword = params.get("keyword")
        if keyword is not None and set(params) & set(_SCRIPT_PARAMS):
            raise ValueError("trigger_mode: a keyword's line has no script parameters to compare")
        if keyword is None and "mode" not in params:
            raise ValueError("trigger_mode: a declaration names the trigger's mode or keyword")
        if "named" in str(params.get("selector", "")).lower():
            raise ValueError("trigger_mode: a selector naming a card fixes one scenario's board")
        if record.kind is not RecordKind.TRIGGER:
            return None
        if keyword is not None:
            if not is_keyword_line(acting_line(record, lines), keyword):
                return None
            return self._side(record)
        found = acting_root_params(record, lines)
        if found is None:
            return None
        for name, script_name in _SCRIPT_PARAMS.items():
            if name in params and found.get(script_name) != params[name]:
                return None
        return self._side(record)

    @staticmethod
    def _side(record: EffectRecord) -> FamilyMatch:
        fired = record.payload.fired
        return FamilyMatch(Polarity.FIRES if fired else Polarity.DOES_NOT_FIRE, fired,
                           tuple(record.payload.event.subjects))


FAMILY = register(TriggerMode())
