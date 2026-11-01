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
  static forbids (``attackers`` and ``blockers`` record who forbids them);
- ``universe``: ``battlefield`` or ``battlefield_and_players``, the entities
  the list is drawn from, so one of them the record does not list counts as
  outside too: the noncreature permanents no attackers record lists (only
  creatures attack, 506.3), the permanent and player a hexproof or an enchant
  restriction keeps out of a candidate's ``legal_targets``. The universe is
  the snapshot's battlefield, for ``legal_attackers`` the active player's
  permanents and for ``legal_blockers`` the other player's, plus the players
  under ``battlefield_and_players``. A static is never recorded for an
  unlisted entity, so ``responsible_static`` leaves them out.

Without ``universe`` only an entity the record lists counts: the legal list's
members are inside, the forbidden entities outside, and a decision record,
which lists no outside, speaks for its ``legal_targets``' inside alone. With
it, a decision record speaks for a candidate that targets, one listing a legal
target or having none; a gen-2 candidate listing 32 targets is skipped, since
the collector capped the list there and its outside is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, PlayabilitySubkind, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register

UNIVERSES = frozenset({"battlefield", "battlefield_and_players"})
#: The gen-2 collector's cap on a candidate's ``legal_targets``.
GEN2_TARGET_CAP = 32

_SUBKIND_OF = {
    "legal_attackers": PlayabilitySubkind.ATTACKERS,
    "legal_blockers": PlayabilitySubkind.BLOCKERS,
    "legal_targets": PlayabilitySubkind.DECISION,
}


def _universe(record: EffectRecord, listed: str, universe: str) -> tuple[str, ...]:
    """The entities the list is drawn from, in snapshot order."""
    state = record.state
    active = state.global_.active
    found = [e.id for e in state.entities if e.on_battlefield and (
        listed == "legal_targets"
        or (e.controller == active) == (listed == "legal_attackers"))]
    if universe == "battlefield_and_players":
        found += [p.id for p in state.players]
    return tuple(found)


def _sides(record: EffectRecord, listed: str, statics_only: bool, universe: str | None):
    payload = record.payload
    if listed == "legal_targets":
        candidates = payload.candidates
        if universe is not None:
            candidates = [c for c in candidates
                          if (c.legal_targets or not c.has_legal_target)
                          and (record.generation == 3 or len(c.legal_targets) < GEN2_TARGET_CAP)]
            if not candidates:
                return (), ()
        inside = tuple(dict.fromkeys(t for c in candidates for t in c.legal_targets))
        outside: tuple[str, ...] = ()
    else:
        inside = tuple(getattr(payload, listed))
        outside = tuple(f.entity for f in payload.forbidden
                        if not statics_only or f.responsible_static)
    if universe is not None and not statics_only:
        listed_ids = set(inside) | set(outside)
        outside += tuple(e for e in _universe(record, listed, universe) if e not in listed_ids)
    return inside, outside


@dataclass(frozen=True, slots=True)
class LegalityMembership:
    id: str = "legality_membership"
    params: frozenset[str] = frozenset({"list", "fires_when", "responsible_static", "universe"})

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
        universe = params.get("universe")
        if universe is not None and universe not in UNIVERSES:
            raise ValueError(f"legality_membership: unknown universe {universe!r}")
        if record.kind is not RecordKind.PLAYABILITY or record.subkind is not _SUBKIND_OF[listed]:
            return None
        inside, outside = _sides(record, listed, bool(params.get("responsible_static")),
                                 universe)
        fires, near = (outside, inside) if fires_when == "outside" else (inside, outside)
        if fires:
            return FamilyMatch(Polarity.FIRES, True, fires)
        if near:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, near)
        return None


FAMILY = register(LegalityMembership())
