"""Detector family ``continuous_contribution``: what a static ability does to the board.

A ``continuous`` record lists, per affected entity, what the acting static
contributes: a P/T boost, keywords, types, colours. A rule about a layer's
effect is on its ``fires`` side when some contribution carries it, and on its
near miss when the static applies to the board and contributes nothing of it.

Parameters:

- ``field`` (required): ``pt``, ``keywords``, ``types`` or ``colors``;
- ``value``: a keyword, type or colour the contribution must name (``flying``,
  ``artifact``, ``U``), as the record spells it; a leading ``-`` names a
  removal, a lone ``=`` a characteristic set outright.

Prior art: ``effect_targets`` derives the P/T, keyword, type and colour
targets of a ``continuous`` record from the same contributions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import SidecarLookup
from effects.domain.records import Contribution, EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register

FIELDS = frozenset({"pt", "keywords", "types", "colors"})


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
    params: frozenset[str] = frozenset({"field", "value"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        field = params.get("field")
        if field not in FIELDS:
            raise ValueError(f"continuous_contribution: unknown field {field!r}")
        if record.kind is not RecordKind.CONTINUOUS:
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
