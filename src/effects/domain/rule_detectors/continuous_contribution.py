"""Detector family ``continuous_contribution``: what a static ability does to the board.

A ``continuous`` record lists, per affected entity, what the acting static
contributes: a P/T boost, keywords, types, colours. A rule about a layer's
effect is on its ``fires`` side when some contribution carries it, and on its
near miss when the static applies to the board and contributes nothing of it.

Parameters:

- ``field`` (required): ``pt``, ``keywords``, ``types`` or ``colors``;
- ``value``: a keyword, type or colour the contribution must name (``flying``,
  ``artifact``, ``U``), compared without case; a leading ``-`` names a
  removal, a lone ``=`` a characteristic set outright;
- ``source_token_kind``: the static's source is a token of this kind (a
  Monster or a Royal Role, rule 111.10), found as ``outcome_event``'s is: a
  ``continuous`` record names no source, so its host is the battlefield entity
  made from or printed by the static's script file;
- ``counter_condition``: the static is conditioned on its host's counters of
  this type (``LEVEL``: a leveler's ``IsPresent$ Card.Self+counters_GE4_LEVEL``),
  read off the condition keys of the static's root segment (``IsPresent$``,
  ``Condition…$``, ``CheckSVar$``), never its ``Affected$`` selector; counter
  types in ``normalize_counter_type``'s spelling;
- ``source_keyword``: the static's card prints this keyword on any of its
  lines (a bestow Aura's enchanted-creature static, a soulbond pair's
  static), read off its sidecar as ``outcome_event`` reads it;
- ``script_param``: the static's root segment holds this parameter, a name or
  a name -> value map as ``outcome_event`` reads it (``{"Condition":
  "MaxSpeed"}``, ``{"IsPresent": "Card.Self+IsSolved"}``);
- ``affected_face_state``: only contributions to an entity in this gen-3 face
  state count, on either side (a flipped permanent, a face-down one);
- ``designation``: the contribution is the work of a gen-3 designation
  (``suspected``): a contribution counts on the ``fires`` side only to an
  entity holding it, and the near miss is the static contributing nothing to
  an entity lacking it.

The condition entity is the static's host (``acting_line.source_ids``: a
``continuous`` record names no source, so the battlefield entity printed by
or made from the static's script file); the outcome entities are the
contributions' entities, those carrying the contribution on the ``fires``
side, every one the static applies to on the near miss.

Prior art: ``effect_targets`` derives the P/T, keyword, type and colour
targets of a ``continuous`` record from the same contributions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import SidecarLookup
from effects.domain.records import Contribution, EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    acting_root_params,
    card_has_keyword,
    script_params_match,
    source_ids,
    source_is_token_of_kind,
)
from effects.domain.rule_detectors.entity_facts import has_designation, has_face_state
from effects.domain.script_text import params, root
from effects.domain.state_snapshot import normalize_counter_type

FIELDS = frozenset({"pt", "keywords", "types", "colors"})

#: A counter count a static's condition reads: ``counters_GE4_LEVEL`` in a
#: selector, ``CardCounters.LEVEL`` in a count.
COUNTER_READ = re.compile(
    r"counters_[A-Z]{2}\w*?_([A-Za-z0-9]+)|CardCounters\.([A-Za-z0-9]+)")

#: The root-segment keys that gate whether a static applies at all. An
#: ``Affected$`` selector reading counters (each creature with a +1/+1 counter
#: has vigilance) says what the static applies to, not whether it is on.
_CONDITION_KEYS = ("IsPresent", "Condition", "CheckSVar", "CheckSecondSVar")


def condition_counter_types(script_text: str | None) -> frozenset[str]:
    """The counter types a static's condition reads, in ``normalize_counter_type``'s
    spelling; empty for a static no counter count turns on."""
    if not script_text:
        return frozenset()
    gates = [value for key, value in params(root(script_text)).items()
             if key.startswith(_CONDITION_KEYS)]
    return frozenset(normalize_counter_type(a or b)
                     for value in gates for a, b in COUNTER_READ.findall(value))


def conditioned_on_counters(script_text: str | None, counter_type: str) -> bool:
    """Whether a static's condition reads its ``counter_type`` counters."""
    return normalize_counter_type(counter_type) in condition_counter_types(script_text)


def _carries(contribution: Contribution, field: str, value: str | None) -> bool:
    if field == "pt":
        return contribution.pt_boost != (0, 0)
    found = getattr(contribution, field)
    if value is None:
        return bool(found)
    if field == "keywords":
        return normalize_keyword(value) in {normalize_keyword(k) for k in found}
    # The record spells types lower-case (``creature``, ``-creature``), a
    # declaration as the rules do (``Creature``).
    return value.lower() in {f.lower() for f in found}


@dataclass(frozen=True, slots=True)
class ContinuousContribution:
    id: str = "continuous_contribution"
    params: frozenset[str] = frozenset(
        {"field", "value", "source_token_kind", "counter_condition", "source_keyword",
         "script_param", "affected_face_state", "designation"})
    general: frozenset[str] = frozenset({"field", "value", "source_keyword", "designation"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        needs = set()
        if "affected_face_state" in params:
            needs.add("face_state")
        if "designation" in params:
            needs.add("designations")
        return frozenset(needs)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        field = params.get("field")
        if field not in FIELDS:
            raise ValueError(f"continuous_contribution: unknown field {field!r}")
        if record.kind is not RecordKind.CONTINUOUS:
            return None
        kind = params.get("source_token_kind")
        if kind is not None and not source_is_token_of_kind(record.state, kind, record.ability):
            return None
        counters = params.get("counter_condition")
        if counters is not None:
            line = acting_line(record, lines)
            if not conditioned_on_counters(line.script_text if line else None, counters):
                return None
        if "script_param" in params and not script_params_match(
                acting_root_params(record, lines), params["script_param"]):
            return None
        if "source_keyword" in params and not card_has_keyword(
                record.ability or (), lines, params["source_keyword"]):
            return None
        contributions = record.payload.contributions
        face = params.get("affected_face_state")
        if face is not None:
            contributions = tuple(c for c in contributions if _entity_fact(
                record, c.entity, lambda entity: has_face_state(entity, face)))
        if not contributions:
            return None
        carrying = tuple(dict.fromkeys(
            c.entity for c in contributions if _carries(c, field, params.get("value"))))
        applied = tuple(dict.fromkeys(c.entity for c in contributions))
        designation = params.get("designation")
        if designation is not None:
            def holds(entity_id: str) -> bool | None:
                return _entity_fact(record, entity_id,
                                    lambda entity: has_designation(entity, designation))
            applied = tuple(e for e in applied if e not in carrying and holds(e) is False)
            carrying = tuple(e for e in carrying if holds(e))
        host = source_ids(record)
        if carrying:
            return FamilyMatch(Polarity.FIRES, True, host, carrying)
        if not applied:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, host, applied)


def _entity_fact(record: EffectRecord, entity_id: str, read) -> bool | None:
    """``read`` of the snapshot entity ``entity_id``; None where it is not in
    the snapshot or the fact is unreadable."""
    entity = record.state.entity(entity_id)
    return None if entity is None else read(entity)


FAMILY = register(ContinuousContribution())
