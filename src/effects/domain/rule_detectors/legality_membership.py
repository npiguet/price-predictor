"""Detector family ``legality_membership``: an entity inside or outside a legality list.

Legality records list who may attack, who may block an anchored attacker, and
which entities a candidate may target; the ``attackers`` and ``blockers``
subkinds also list the entities a static forbids. A legality rule is on its
``fires`` side for an entity the rule puts on the side ``fires_when`` names
(``outside`` by default: "can't block", "can't be the target of"), and on its
near miss for an entity on the other side.

Parameters:

- ``list`` (required): ``legal_attackers``, ``legal_blockers`` or
  ``legal_targets``;
- ``fires_when``: ``outside`` (default) or ``inside``;
- ``responsible_static``: true to count, on the outside, only entities a
  static forbids (``attackers`` and ``blockers`` record who forbids them).

Only an entity the record lists counts: the legal list's members are inside,
the forbidden entities outside. A decision record lists no outside, so its
``legal_targets`` speak only for the inside.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, PlayabilitySubkind, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register

_SUBKIND_OF = {
    "legal_attackers": PlayabilitySubkind.ATTACKERS,
    "legal_blockers": PlayabilitySubkind.BLOCKERS,
    "legal_targets": PlayabilitySubkind.DECISION,
}


def _sides(record: EffectRecord, listed: str, statics_only: bool):
    payload = record.payload
    if listed == "legal_targets":
        inside = tuple(dict.fromkeys(t for c in payload.candidates for t in c.legal_targets))
        return inside, ()
    inside = tuple(getattr(payload, listed))
    outside = tuple(f.entity for f in payload.forbidden
                    if not statics_only or f.responsible_static)
    return inside, outside


@dataclass(frozen=True, slots=True)
class LegalityMembership:
    id: str = "legality_membership"
    params: frozenset[str] = frozenset({"list", "fires_when", "responsible_static"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        listed = params.get("list")
        if listed not in _SUBKIND_OF:
            raise ValueError(f"legality_membership: unknown list {listed!r}")
        fires_when = params.get("fires_when", "outside")
        if fires_when not in ("inside", "outside"):
            raise ValueError(f"legality_membership: fires_when {fires_when!r}")
        if record.kind is not RecordKind.PLAYABILITY or record.subkind is not _SUBKIND_OF[listed]:
            return None
        inside, outside = _sides(record, listed, bool(params.get("responsible_static")))
        fires, near = (outside, inside) if fires_when == "outside" else (inside, outside)
        if fires:
            return FamilyMatch(Polarity.FIRES, True, fires)
        if near:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, near)
        return None


FAMILY = register(LegalityMembership())
