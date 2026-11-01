"""Detector family ``continuous_contribution``: what a static ability does to the board.

A ``continuous`` record lists, per affected entity, what the acting static
contributes: a P/T boost, keywords, types, colours. A rule about a layer's
effect is on its ``fires`` side when some contribution carries it, and on its
near miss when the static applies to the board and contributes nothing of it.

Parameters:

- ``field`` (required): ``pt``, ``keywords``, ``types`` or ``colors``;
- ``value``: a keyword, type or colour the contribution must name (``flying``,
  ``artifact``, ``U``), as the record spells it; a leading ``-`` names a
  removal, a lone ``=`` a characteristic set outright;
- ``source_token_kind``: the static's source is a token of this kind (a
  Monster or a Royal Role, rule 111.10), found as ``outcome_event``'s is: a
  ``continuous`` record names no source, so its host is the battlefield entity
  made from or printed by the static's script file;
- ``counter_condition``: the static is conditioned on its host's counters of
  this type (``LEVEL``: a leveler's ``IsPresent$ Card.Self+counters_GE4_LEVEL``),
  read off the condition keys of the static's root segment (``IsPresent$``,
  ``Condition…$``, ``CheckSVar$``), never its ``Affected$`` selector; counter
  types in ``normalize_counter_type``'s spelling.

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
from effects.domain.rule_detectors.acting_line import acting_line, source_is_token_of_kind
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
    return value in found


@dataclass(frozen=True, slots=True)
class ContinuousContribution:
    id: str = "continuous_contribution"
    params: frozenset[str] = frozenset(
        {"field", "value", "source_token_kind", "counter_condition"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

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
        contributions = record.payload.contributions
        if not contributions:
            return None
        carrying = tuple(dict.fromkeys(
            c.entity for c in contributions if _carries(c, field, params.get("value"))))
        if carrying:
            return FamilyMatch(Polarity.FIRES, True, carrying)
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False,
                           tuple(dict.fromkeys(c.entity for c in contributions)))


FAMILY = register(ContinuousContribution())
