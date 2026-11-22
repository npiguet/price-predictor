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
suffice; a blockers record writes it.

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
- ``restriction`` (``legal_blockers`` and ``legal_targets``): the one rule
  that keeps an entity out, decided by ``rule_outcomes``
  (``legality_restrictions``: an evasion, ``tapped``, ``landwalk`` or
  ``effect`` for a blocker; ``hexproof``, ``hexproof_from``, ``shroud``,
  ``protection`` or ``selector`` for a target), with ``quality`` (a
  protection's quality kind, ``everything``) beside ``protection``;
- ``judged`` (``permanent`` or ``player``), ``source_keyword`` (the
  candidate's card prints this keyword), ``candidate`` (``cast`` or
  ``activated``: the candidate is its card's cast or an activated ability,
  which a ``source_keyword`` always names) and ``candidate_keyword`` (the
  candidate is this keyword's line), on ``legal_targets``;
- ``designation``: only a judged entity holding this gen-3 designation counts
  (a suspected creature);
- ``judged_not_type``: only a judged entity without this card type counts,
  on either side (a noncreature permanent, 506.3);
- ``restriction`` on ``legal_attackers``: ``tapped`` or ``sick`` (summoning
  sick without haste), 508.1a's two conditions, judged on a gen-3 record's
  anchor alone;
- ``min_types``: only a judged entity with at least this many card types
  counts, on either side (an artifact creature, 205.2b).

Without a ``restriction``, a near miss is counted only under one of the
parameters that narrow the judged entities (``SCOPES``: ``keyword``,
``counter_type``, ``judged_type``, ``judged_not_type``, ``min_types``,
``designation``); otherwise the list's other side is every entity the rule
says nothing of.

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
target or having none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.damage_step_keywords import KeywordsOf
from effects.domain.effect_head_layout import HeadField
from effects.domain.effect_targets import derive_targets
from effects.domain.provenance import SidecarLookup
from effects.domain.records import Candidate, EffectRecord, PlayabilitySubkind, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import (
    EventlessFamily,
    FamilyMatch,
    Matcher,
    first_side,
    register,
)
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    candidate_kind,
    card_has_keyword,
    is_keyword_line,
    line_of,
    require_candidate,
)
from effects.domain.rule_detectors.entity_facts import (
    card_type_count,
    has_card_type,
    has_designation,
)
from effects.domain.rule_detectors.keyword_reading import KeywordReading
from effects.domain.rule_detectors.legality_restrictions import (
    ATTACKER_RESTRICTIONS,
    BLOCKER_RESTRICTIONS,
    JUDGED,
    QUALITY_KINDS,
    SELECTOR,
    TARGET_RESTRICTIONS,
    Restriction,
    TargetPool,
    attacker_judgements,
    blocker_judgements,
    sides,
    target_judgements,
)
from effects.domain.state_snapshot import EntityState, normalize_counter_type

UNIVERSES = frozenset({"battlefield", "battlefield_and_players"})

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

_Ids = tuple[str, ...]
_Sides = tuple[_Ids, _Ids]


@dataclass(frozen=True, slots=True)
class _Reading:
    """One playability record as the matcher reads it."""

    record: EffectRecord
    lines: SidecarLookup | None
    keywords_of: KeywordsOf


# ── the universe ────────────────────────────────────────────────────────


def _universe(record: EffectRecord, listed: str, universe: str) -> _Ids:
    """The entities the list is drawn from, in snapshot order."""
    state = record.state
    if listed == "legal_defenders":
        return _defender_universe(record, universe)
    active = state.global_.active
    found = [e.id for e in state.entities if _drawn(e, listed, active)]
    if universe == "battlefield_and_players":
        found += [p.id for p in state.players]
    return tuple(found)


def _drawn(entity: EntityState, listed: str, active: str | None) -> bool:
    """A permanent of the list's battlefield: any for ``legal_targets``, the
    active player's for ``legal_attackers``, the other's for ``legal_blockers``."""
    if not entity.on_battlefield:
        return False
    return listed == "legal_targets" or (entity.controller == active) == (
        listed == "legal_attackers")


def _defender_universe(record: EffectRecord, universe: str) -> _Ids:
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


# ── a list's inside and outside ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Listed:
    """``(fires, near)`` by the list's inside and outside alone. The near
    miss is counted only under a scoping parameter (``SCOPES``): without
    one, it is every entity on the list's other side, every creature that may
    attack for a can't-attack rule, every attacker no requirement binds."""

    listed: str
    fires_outside: bool
    statics_only: bool
    universe: str | None
    at_least: int
    scoped: bool

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> _Listed:
        return cls(params.get("list"), params.get("fires_when", "outside") == "outside",
                   bool(params.get("responsible_static")), params.get("universe"),
                   int(params.get("at_least", 2)), bool(SCOPES & set(params)))

    def __call__(self, reading: _Reading) -> _Sides:
        inside, outside = _LIST_SIDES[self.listed](self, reading.record)
        fires, near = (outside, inside) if self.fires_outside else (inside, outside)
        return fires, near if self.scoped else ()

    def anchor_sides(self, record: EffectRecord) -> _Sides:
        """The anchor attacker inside or outside an anchor list."""
        payload = record.payload
        anchor = payload.anchor_attacker
        if anchor is None:
            return (), ()
        inside = (payload.must_attack if self.listed == "must_attack"
                  else payload.min_blockers >= self.at_least)
        return ((anchor,), ()) if inside else ((), (anchor,))

    def must_block_sides(self, record: EffectRecord) -> _Sides:
        """The creatures a requirement makes block the anchor, and the other listed ones."""
        payload = record.payload
        if payload.must_block is None:
            return (), ()
        bound = set(payload.must_block)
        listed_ids = (*payload.legal_blockers, *(f.entity for f in payload.forbidden))
        return (tuple(payload.must_block),
                tuple(e for e in dict.fromkeys(listed_ids) if e not in bound))

    def defender_sides(self, record: EffectRecord) -> _Sides:
        """The anchor's legal defenders, and outside them the universe's others."""
        payload = record.payload
        if payload.anchor_attacker is None:
            return (), ()
        inside = payload.legal_defenders
        outside: _Ids = ()
        if self.universe is not None:
            outside = tuple(e for e in _universe(record, self.listed, self.universe)
                            if e not in inside)
        return inside, outside

    def target_sides(self, record: EffectRecord) -> _Sides:
        """The legal targets of the candidates that target; nothing is listed outside."""
        candidates = record.payload.candidates
        if self.universe is not None:
            candidates = [c for c in candidates if _targets(c)]
            if not candidates:
                return (), ()
        inside = tuple(dict.fromkeys(t for c in candidates for t in c.legal_targets))
        return self._with_universe(record, inside, ())

    def blocker_sides(self, record: EffectRecord) -> _Sides:
        return self._with_universe(record, *self._legal_and_forbidden(record))

    def attacker_sides(self, record: EffectRecord) -> _Sides:
        inside, outside = self._legal_and_forbidden(record)
        anchor = getattr(record.payload, "anchor_attacker", None)
        if anchor is None:
            return self._with_universe(record, inside, outside)
        return self._anchored(record, anchor, inside, outside)

    def _legal_and_forbidden(self, record: EffectRecord) -> _Sides:
        """The list's members, and the forbidden entities (those a static
        forbids alone under ``responsible_static``)."""
        payload = record.payload
        return (tuple(getattr(payload, self.listed)),
                tuple(f.entity for f in payload.forbidden
                      if not self.statics_only or f.responsible_static))

    def _anchored(self, record: EffectRecord, anchor: str, inside: _Ids,
                  outside: _Ids) -> _Sides:
        """A gen-3 declaration writes one record per creature, each listing
        the declaration's whole lists: each judges its own anchor, and the
        universe's unlisted permanents are judged once, on the record whose
        anchor the lists name first."""
        listed_ids = (*inside, *(f.entity for f in record.payload.forbidden))
        first = bool(listed_ids) and listed_ids[0] == anchor
        if first and self._draws_universe():
            outside += self._unlisted(record, set(listed_ids))
        return (tuple(e for e in inside if e == anchor),
                tuple(e for e in outside if _judged_here(e, anchor, listed_ids, first)))

    def _with_universe(self, record: EffectRecord, inside: _Ids, outside: _Ids) -> _Sides:
        """The sides, the universe's unlisted entities outside where the
        declaration draws one."""
        if not self._draws_universe():
            return inside, outside
        return inside, outside + self._unlisted(record, set(inside) | set(outside))

    def _draws_universe(self) -> bool:
        """A static is never recorded for an unlisted entity, so
        ``responsible_static`` draws no universe."""
        return self.universe is not None and not self.statics_only

    def _unlisted(self, record: EffectRecord, known: set[str]) -> _Ids:
        return tuple(e for e in _universe(record, self.listed, self.universe) if e not in known)


def _targets(candidate: Candidate) -> bool:
    """A candidate that targets: one listing a legal target or having none."""
    return bool(candidate.legal_targets) or not candidate.has_legal_target


def _judged_here(entity_id: str, anchor: str, listed_ids: tuple[str, ...], first: bool) -> bool:
    return entity_id == anchor or (entity_id not in listed_ids and first)


#: How each list reads its inside and outside.
_LIST_SIDES: dict[str, Callable[[_Listed, EffectRecord], _Sides]] = {
    "must_attack": _Listed.anchor_sides,
    "min_blockers": _Listed.anchor_sides,
    "must_block": _Listed.must_block_sides,
    "legal_defenders": _Listed.defender_sides,
    "legal_targets": _Listed.target_sides,
    "legal_attackers": _Listed.attacker_sides,
    "legal_blockers": _Listed.blocker_sides,
}


# ── a named restriction ─────────────────────────────────────────────────


#: The restrictions each list can name (``legality_restrictions``).
_RESTRICTIONS = {"legal_blockers": BLOCKER_RESTRICTIONS, "legal_targets": TARGET_RESTRICTIONS,
                 "legal_attackers": ATTACKER_RESTRICTIONS}
#: The parameters that read the candidates of a targeting decision.
_READS_CANDIDATES = frozenset({"restriction", "source_keyword", "candidate_keyword", "candidate"})

#: A candidate filter's test with its declared value bound.
_CandidateTest = Callable[[Candidate, "SidecarLookup | None"], bool]


@dataclass(frozen=True, slots=True)
class _SourceKeyword:
    keyword: str

    def __call__(self, candidate: Candidate, lines: SidecarLookup | None) -> bool:
        return card_has_keyword(candidate.ability, lines, self.keyword)


@dataclass(frozen=True, slots=True)
class _CandidateKind:
    kind: str

    def __call__(self, candidate: Candidate, lines: SidecarLookup | None) -> bool:
        return candidate_kind(candidate, lines) == self.kind


@dataclass(frozen=True, slots=True)
class _CandidateKeyword:
    keyword: str

    def __call__(self, candidate: Candidate, lines: SidecarLookup | None) -> bool:
        return is_keyword_line(line_of(candidate.ability, lines), self.keyword)


#: The candidate filters, in the order they are read, each from its parameter.
_CANDIDATE_TESTS: tuple[tuple[str, Callable[[Any], _CandidateTest]], ...] = (
    ("source_keyword", _SourceKeyword),
    ("candidate", _CandidateKind),
    ("candidate_keyword", _CandidateKeyword),
)


@dataclass(frozen=True, slots=True)
class _Restricted:
    """``(fires, near)`` under a named restriction (``legality_restrictions``)."""

    listed: str
    restriction: Restriction
    fires_when: str
    candidate_tests: tuple[_CandidateTest, ...]
    #: ``judged`` on ``legal_targets``: players or permanents alone.
    players: bool | None

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> _Restricted:
        judged = params.get("judged")
        return cls(params["list"],
                   Restriction(params.get("restriction", SELECTOR), params.get("quality")),
                   params.get("fires_when", "outside"),
                   tuple(make(params[name]) for name, make in _CANDIDATE_TESTS if name in params),
                   None if judged is None else judged == "player")

    def __call__(self, reading: _Reading) -> _Sides:
        record, keywords_of = reading.record, reading.keywords_of
        if self.listed == "legal_blockers":
            judged = blocker_judgements(record, keywords_of, self.restriction)
        elif self.listed == "legal_attackers":
            judged = attacker_judgements(record, keywords_of, self.restriction.name)
        else:
            judged = target_judgements(record, keywords_of, self.restriction, self._pool(reading))
        return sides(judged, self.fires_when)

    def _pool(self, reading: _Reading) -> TargetPool:
        record = reading.record
        candidates = [c for c in record.payload.candidates
                      if _targets(c) and self._speaks_for(c, reading.lines)]
        universe = _universe(record, "legal_targets", "battlefield_and_players")
        if self.players is not None:
            players = {p.id for p in record.state.players}
            universe = tuple(e for e in universe if (e in players) == self.players)
        return TargetPool(candidates, universe)

    def _speaks_for(self, candidate: Candidate, lines: SidecarLookup | None) -> bool:
        return all(test(candidate, lines) for test in self.candidate_tests)


# ── the judged entities' own filters ────────────────────────────────────


#: A filter of the judged entities with its declared value bound: whether an
#: entity on the ``fires`` side (``True``) or the near miss counts.
_EntityTest = Callable[[EntityState, "SidecarLookup | None", bool], bool]


@dataclass(frozen=True, slots=True)
class _PrintsKeyword:
    keyword: str

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        return card_has_keyword(entity.printed, lines, self.keyword)


@dataclass(frozen=True, slots=True)
class _HoldsCounter:
    """On the ``fires`` side an entity holding the counter, on the near miss one without."""

    counter: str

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        held = any(normalize_counter_type(name) == self.counter and n > 0
                   for name, n in entity.counters.items())
        return held == fires


@dataclass(frozen=True, slots=True)
class _HoldsDesignation:
    designation: str

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        return bool(has_designation(entity, self.designation))


@dataclass(frozen=True, slots=True)
class _LacksType:
    card_type: str

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        return not has_card_type(entity, self.card_type)


@dataclass(frozen=True, slots=True)
class _OfType:
    card_type: str

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        return has_card_type(entity, self.card_type)


@dataclass(frozen=True, slots=True)
class _MinTypes:
    count: int

    def __call__(self, entity: EntityState, lines: SidecarLookup | None, fires: bool) -> bool:
        return card_type_count(entity) >= self.count


def _counter_test(value: Any) -> _EntityTest:
    return _HoldsCounter(normalize_counter_type(value))


def _min_types_test(value: Any) -> _EntityTest:
    return _MinTypes(int(value))


def _given(params: Mapping[str, Any], name: str) -> bool:
    return params.get(name) is not None


def _named(params: Mapping[str, Any], name: str) -> bool:
    return name in params


#: The judged entities' filters, in the order they are read, each from its
#: parameter, read where the declaration gives it a value (``_given``) or
#: names it (``_named``).
_ENTITY_TESTS: tuple[tuple[str, Callable[[Mapping[str, Any], str], bool],
                           Callable[[Any], _EntityTest]], ...] = (
    ("keyword", _given, _PrintsKeyword),
    ("counter_type", _given, _counter_test),
    ("designation", _named, _HoldsDesignation),
    ("judged_not_type", _named, _LacksType),
    ("judged_type", _given, _OfType),
    ("min_types", _given, _min_types_test),
)


def _entity_tests(params: Mapping[str, Any]) -> tuple[_EntityTest, ...]:
    return tuple(make(params[name]) for name, read, make in _ENTITY_TESTS
                 if read(params, name))


# ── the matcher ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, eq=False)
class MembershipMatcher:
    """A compiled ``legality_membership`` declaration."""

    listed: str
    subkind: PlayabilitySubkind
    sides: Callable[[_Reading], _Sides]
    tests: tuple[_EntityTest, ...]
    keywords: KeywordReading = field(default_factory=KeywordReading)

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> MembershipMatcher:
        listed = params.get("list")
        reads = _Restricted.of if _READS_CANDIDATES & params.keys() else _Listed.of
        return cls(listed, _SUBKIND_OF[listed], reads(params), _entity_tests(params))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if record.kind is not RecordKind.PLAYABILITY or record.subkind is not self.subkind:
            return None
        reading = _Reading(record, lines, self.keywords.of(lines).keywords_of)
        fires, near = self.sides(reading)
        side = first_side(self._kept(reading, fires, True), self._kept(reading, near, False))
        if side is None:
            return None
        polarity, judged = side
        condition, outcome = _roles(record, self.listed, judged)
        return FamilyMatch(polarity, polarity is Polarity.FIRES, condition, outcome)

    def _kept(self, reading: _Reading, ids: _Ids, fires: bool) -> _Ids:
        """The entities of ``ids`` every filter keeps; all of them where the
        declaration filters none."""
        if not self.tests:
            return ids
        return tuple(e for e in ids if self._passes(reading, e, fires))

    def _passes(self, reading: _Reading, entity_id: str, fires: bool) -> bool:
        entity = reading.record.state.entity(entity_id)
        if entity is None:
            return False
        return all(test(entity, reading.lines, fires) for test in self.tests)


# ── the family ──────────────────────────────────────────────────────────


def _check_list(listed: Any, params: Mapping[str, Any]) -> None:
    if listed not in _SUBKIND_OF:
        raise ValueError(f"legality_membership: unknown list {listed!r}")
    fires_when = params.get("fires_when", "outside")
    if fires_when not in ("inside", "outside"):
        raise ValueError(f"legality_membership: fires_when {fires_when!r}")
    _check_universe(listed, params)


def _check_universe(listed: Any, params: Mapping[str, Any]) -> None:
    universe = params.get("universe")
    if universe is not None and universe not in UNIVERSES:
        raise ValueError(f"legality_membership: unknown universe {universe!r}")
    if "at_least" in params and listed != "min_blockers":
        raise ValueError("legality_membership: at_least reads the min_blockers list")


def _check_restriction(listed: str, params: Mapping[str, Any]) -> None:
    restriction = params.get("restriction")
    if restriction is not None:
        known = _RESTRICTIONS.get(listed, frozenset())
        if restriction not in known:
            raise ValueError(f"legality_membership: restriction {restriction!r} on "
                             f"{listed} is none of {sorted(known)}")
        if params.get("responsible_static"):
            raise ValueError("legality_membership: a restriction names the rule; "
                             "responsible_static reads no rule")
    if not _quality_applies(params, restriction):
        raise ValueError(f"legality_membership: quality is one of {sorted(QUALITY_KINDS)}, "
                         "with restriction protection")


def _quality_applies(params: Mapping[str, Any], restriction: Any) -> bool:
    if "quality" not in params:
        return True
    return restriction == "protection" and params["quality"] in QUALITY_KINDS


def _check_targets_only(listed: str, params: Mapping[str, Any]) -> None:
    for name in ("judged", "source_keyword", "candidate_keyword", "candidate"):
        if name in params and listed != "legal_targets":
            raise ValueError(f"legality_membership: {name} reads legal_targets")
    if params.get("judged", "permanent") not in JUDGED:
        raise ValueError(f"legality_membership: judged is one of {sorted(JUDGED)}")


_VALUE_CHECKS = {
    "responsible_static": dv.boolean, "keyword": dv.keyword,
    "source_keyword": dv.keyword, "candidate_keyword": dv.keyword,
    "counter_type": dv.counter_type, "at_least": dv.integer, "min_types": dv.integer,
    "judged_type": dv.card_type, "judged_not_type": dv.card_type,
    "designation": dv.designation}


@dataclass(frozen=True, slots=True)
class LegalityMembership(EventlessFamily):
    id: str = "legality_membership"
    kinds: frozenset[str] = frozenset({"playability"})
    params: frozenset[str] = frozenset({"list", "fires_when", "responsible_static", "universe",
                                        "keyword", "counter_type", "at_least", "judged_type",
                                        "min_types", "restriction", "quality", "judged",
                                        "source_keyword", "candidate_keyword", "designation",
                                        "judged_not_type", "candidate"})
    general: frozenset[str] = frozenset({"list", "fires_when", "universe", "keyword",
                                         "counter_type", "at_least", "restriction",
                                         "source_keyword", "candidate_keyword"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        listed = params.get("list")
        found = {listed} if listed in GEN3_LISTS else set()
        if "designation" in params:
            found.add("designations")
        return frozenset(found)

    def validate(self, params: Mapping[str, Any]) -> None:
        listed = params.get("list")
        _check_list(listed, params)
        _check_restriction(listed, params)
        require_candidate(self.id, params)
        _check_targets_only(listed, params)
        dv.check(self.id, params, _VALUE_CHECKS)

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return MembershipMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


#: The parameters that say which judged entities a list rule speaks for, so
#: which of them its near miss is; a ``restriction`` decides its own
#: (``legality_restrictions``).
SCOPES = frozenset({"keyword", "counter_type", "judged_type", "judged_not_type", "min_types",
                    "designation"})


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
