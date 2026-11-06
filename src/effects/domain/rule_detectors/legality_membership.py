"""Detector family ``legality_membership``: an entity inside or outside a legality list.

Legality records list who may attack, who may block an anchored attacker, and
which entities a candidate may target; the ``attackers`` and ``blockers``
subkinds also list the entities a static forbids. A legality rule is on its
``fires`` side for an entity the rule puts on the side ``fires_when`` names
(``outside`` by default: "can't block", "can't be the target of"), and on its
near miss for an entity on the other side.

Gen-3 records add two lists. An anchored attackers record's
``legal_defenders`` are the players, planeswalkers and battles its anchor may
attack, and a defender outside them is one it may not ("can't attack
planeswalkers", 506.3); it lists no forbidden defender, so its outside comes
from ``universe`` alone. A blockers record's ``must_block`` is the listed
creatures a requirement makes block the anchor, and its outside the listed
creatures, legal or forbidden, no requirement binds. Both read nothing on an
earlier record.

Two lists judge the anchor attacker alone. ``must_attack`` holds it where an
attack requirement applies to it (gen-3's anchored attackers record: goad,
"attacks each combat if able"), and leaves it outside where none does.
``min_blockers`` holds it where it can be blocked only by ``at_least``
creatures or more (menace, 702.111b), and leaves it outside where fewer
suffice; a blockers record of any generation writes it.

Parameters:

- ``list`` (required): ``legal_attackers``, ``legal_blockers``,
  ``legal_targets``, ``min_blockers``, or gen-3's ``legal_defenders``,
  ``must_block`` and ``must_attack``;
- ``at_least``: with ``list = min_blockers``, the fewest blockers the anchor
  must need to be inside, 2 by default;
- ``fires_when``: ``outside`` (default) or ``inside``;
- ``responsible_static``: true to count, on the outside, only entities a
  static forbids (``attackers`` and ``blockers`` record who forbids them);
- ``universe``: ``battlefield`` or ``battlefield_and_players``, the entities
  the list is drawn from, so one of them the record does not list counts as
  outside too: the noncreature permanents no attackers record lists (only
  creatures attack, 506.3), the permanent and player a hexproof or an enchant
  restriction keeps out of a candidate's ``legal_targets``. The universe is
  the snapshot's battlefield, for ``legal_attackers`` the active player's
  permanents, for ``legal_blockers`` the other player's and for
  ``legal_defenders`` every permanent but the anchor, plus the players under
  ``battlefield_and_players`` (for ``legal_defenders`` those other than the
  anchor's controller). A static is never recorded for an unlisted entity, so
  ``responsible_static`` leaves them out; ``must_block`` takes no universe,
  its outside being listed already;
- ``keyword``: only a listed entity whose card prints this keyword counts, on
  either side, read off its sidecar as ``outcome_event``'s ``source_keyword``
  is: the restriction a keyword makes, on the creatures carrying it (unleash's
  "can't block");
- ``counter_type``: the restriction holds while the entity has a counter of
  this type (unleash's +1/+1 counter), so an entity on the ``fires`` side
  counts only holding one and an entity on the near-miss side only holding
  none, in ``normalize_counter_type``'s spelling;
- ``judged_type``: only a judged entity of this card type counts, on either
  side (a battle among an attacker's defenders), compared without case;
- ``min_types``: only a judged entity with at least this many card types
  counts, on either side (an artifact creature, 205.2b).

Both roles name the judged entities, each one's own characteristics being
what the restriction reads and its legality bit where the outcome falls; a
blockers or defenders record's anchor attacker is a condition entity too
(menace, a "can't be blocked by" ability, live on it), and on a blockers
record an outcome entity, ``min_blockers`` being written there. An outcome
entity is one the record writes the list's legality field at
(``effect_targets``): a universe entity no list names, such as a
noncreature permanent no attackers record lists, has no ``attacker_legal``
target and is a condition entity only.

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

from effects.domain.effect_head_layout import HeadField
from effects.domain.effect_targets import derive_targets
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, PlayabilitySubkind, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import card_has_keyword
from effects.domain.rule_detectors.entity_facts import card_type_count, has_card_type
from effects.domain.state_snapshot import normalize_counter_type

UNIVERSES = frozenset({"battlefield", "battlefield_and_players"})
#: The gen-2 collector's cap on a candidate's ``legal_targets``.
GEN2_TARGET_CAP = 32

_SUBKIND_OF = {
    "legal_attackers": PlayabilitySubkind.ATTACKERS,
    "legal_blockers": PlayabilitySubkind.BLOCKERS,
    "legal_targets": PlayabilitySubkind.DECISION,
    "legal_defenders": PlayabilitySubkind.ATTACKERS,
    "must_block": PlayabilitySubkind.BLOCKERS,
    "must_attack": PlayabilitySubkind.ATTACKERS,
    "min_blockers": PlayabilitySubkind.BLOCKERS,
}
#: The lists only a gen-3 record carries, each the field it reads.
GEN3_LISTS = frozenset({"legal_defenders", "must_block", "must_attack"})
#: The lists that judge the anchor attacker alone.
_ANCHOR_LISTS = frozenset({"must_attack", "min_blockers"})
#: The legality field each list's verdict is written to.
_FIELD_OF = {
    "legal_attackers": HeadField.ATTACKER_LEGAL,
    "legal_blockers": HeadField.BLOCKER_LEGAL,
    "legal_targets": HeadField.TARGET_LEGAL,
    "legal_defenders": HeadField.ATTACK_TARGET_LEGAL,
    "must_block": HeadField.COMBAT_REQUIREMENT,
    "must_attack": HeadField.COMBAT_REQUIREMENT,
    "min_blockers": HeadField.MIN_BLOCKERS,
}


def _universe(record: EffectRecord, listed: str, universe: str) -> tuple[str, ...]:
    """The entities the list is drawn from, in snapshot order."""
    state = record.state
    if listed == "legal_defenders":
        return _defender_universe(record, universe)
    active = state.global_.active
    found = [e.id for e in state.entities if e.on_battlefield and (
        listed == "legal_targets"
        or (e.controller == active) == (listed == "legal_attackers"))]
    if universe == "battlefield_and_players":
        found += [p.id for p in state.players]
    return tuple(found)


def _defender_universe(record: EffectRecord, universe: str) -> tuple[str, ...]:
    """What the anchor could be asked to attack: every other permanent, and
    under ``battlefield_and_players`` every player but its controller."""
    state = record.state
    anchor_id = record.payload.anchor_attacker
    anchor = state.entity(anchor_id)
    found = [e.id for e in state.entities if e.on_battlefield and e.id != anchor_id]
    if universe == "battlefield_and_players":
        attacking = anchor.controller if anchor is not None else state.global_.active
        found += [p.id for p in state.players if p.id != attacking]
    return tuple(found)


def _anchor_sides(record: EffectRecord, listed: str, at_least: int):
    """The anchor attacker inside or outside an anchor list."""
    payload = record.payload
    anchor = payload.anchor_attacker
    if anchor is None or (listed == "must_attack" and record.generation != 3):
        return (), ()
    inside = (payload.must_attack if listed == "must_attack"
              else payload.min_blockers >= at_least)
    return ((anchor,), ()) if inside else ((), (anchor,))


def _gen3_sides(record: EffectRecord, listed: str, universe: str | None):
    payload = record.payload
    if listed == "must_block":
        if payload.must_block is None:
            return None
        bound = set(payload.must_block)
        listed_ids = (*payload.legal_blockers, *(f.entity for f in payload.forbidden))
        return (tuple(payload.must_block),
                tuple(e for e in dict.fromkeys(listed_ids) if e not in bound))
    if payload.anchor_attacker is None:
        return None
    inside = payload.legal_defenders
    outside: tuple[str, ...] = ()
    if universe is not None:
        outside = tuple(e for e in _universe(record, listed, universe) if e not in inside)
    return inside, outside


def _sides(record: EffectRecord, listed: str, statics_only: bool, universe: str | None,
           at_least: int = 2):
    payload = record.payload
    if listed in _ANCHOR_LISTS:
        return _anchor_sides(record, listed, at_least)
    if listed in GEN3_LISTS:
        return _gen3_sides(record, listed, universe) or ((), ())
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


def _of_kind(record: EffectRecord, ids: tuple[str, ...], entity_type: str | None,
             min_types: int | None) -> tuple[str, ...]:
    """The entities of ``ids`` of the card type ``entity_type`` and with at
    least ``min_types`` card types, each filter applying where given."""
    def kept(entity_id: str) -> bool:
        entity = record.state.entity(entity_id)
        if entity is None:
            return False
        if entity_type is not None and not has_card_type(entity, entity_type):
            return False
        return min_types is None or card_type_count(entity) >= min_types
    return tuple(e for e in ids if kept(e))


def _carriers(record: EffectRecord, ids: tuple[str, ...], lines: SidecarLookup | None,
              keyword: str | None, counter: str | None, holding: bool) -> tuple[str, ...]:
    """The entities of ``ids`` whose card prints ``keyword`` and that hold
    (``holding``) or lack a ``counter``, each filter applying where given."""
    def kept(entity_id: str) -> bool:
        entity = record.state.entity(entity_id)
        if entity is None:
            return False
        if keyword is not None and not card_has_keyword(entity.printed, lines, keyword):
            return False
        if counter is None:
            return True
        held = any(normalize_counter_type(name) == counter and n > 0
                   for name, n in entity.counters.items())
        return held == holding
    return tuple(e for e in ids if kept(e))


@dataclass(frozen=True, slots=True)
class LegalityMembership:
    id: str = "legality_membership"
    params: frozenset[str] = frozenset({"list", "fires_when", "responsible_static", "universe",
                                        "keyword", "counter_type", "at_least", "judged_type",
                                        "min_types"})
    general: frozenset[str] = frozenset({"list", "fires_when", "universe", "keyword",
                                         "counter_type", "at_least"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        listed = params.get("list")
        return frozenset({listed}) if listed in GEN3_LISTS else frozenset()

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
        if "at_least" in params and listed != "min_blockers":
            raise ValueError("legality_membership: at_least reads the min_blockers list")
        inside, outside = _sides(record, listed, bool(params.get("responsible_static")),
                                 universe, int(params.get("at_least", 2)))
        fires, near = (outside, inside) if fires_when == "outside" else (inside, outside)
        keyword, counter = params.get("keyword"), params.get("counter_type")
        if keyword is not None or counter is not None:
            counter = None if counter is None else normalize_counter_type(counter)
            fires = _carriers(record, fires, lines, keyword, counter, holding=True)
            near = _carriers(record, near, lines, keyword, counter, holding=False)
        entity_type, min_types = params.get("judged_type"), params.get("min_types")
        if entity_type is not None or min_types is not None:
            min_types = None if min_types is None else int(min_types)
            fires = _of_kind(record, fires, entity_type, min_types)
            near = _of_kind(record, near, entity_type, min_types)
        for polarity, judged in ((Polarity.FIRES, fires), (Polarity.DOES_NOT_FIRE, near)):
            if judged:
                condition, outcome = _roles(record, listed, judged)
                return FamilyMatch(polarity, polarity is Polarity.FIRES, condition, outcome)
        return None


def _roles(record: EffectRecord, listed: str, judged: tuple[str, ...]):
    """The judged entities as condition entities, and those the record
    writes the list's field at as outcome entities, each beside the anchor
    attacker where the record has one."""
    payload = record.payload
    anchor = getattr(payload, "anchor_attacker", None)
    targets = derive_targets(record)
    name = _FIELD_OF[listed]
    outcome = [e for e in judged if (entry := targets.get(e)) is not None
               and name in entry.fields]
    condition = list(judged)
    if anchor:
        condition.append(anchor)
        if record.subkind is PlayabilitySubkind.BLOCKERS:
            outcome.append(anchor)
    return tuple(dict.fromkeys(condition)), tuple(dict.fromkeys(outcome))


FAMILY = register(LegalityMembership())
