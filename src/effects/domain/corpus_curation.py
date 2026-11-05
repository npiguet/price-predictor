"""Which records a curated training corpus keeps, and how many times.

Every decision is shaped so a write worker applies it to one shard alone, by
comparing a record's hash with a :class:`CopyPlan` threshold; all of it is a
pure function of the corpus, the flags and the seed (FR-052). A text over its
cap contributes a random subset rather than its first N, which would be its
earliest games' opening boards over and over.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
from effects.domain.effect_targets import derive_targets
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import (
    Candidate,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityCheck,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.sampling import SamplingClass
from effects.domain.script_text import params, root, segments
from effects.domain.seeded_draw import id_hash, keyed_hash


def record_hash(record_id: str, *, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for one record.

    Not the built-in ``hash()``, which is salted per process, so the survey and
    write passes would silently admit different records (FR-088a).
    """
    return keyed_hash(record_id, seed)


class _BoundedHeap:
    """The ``cap`` largest entries pushed, as a min-heap: what both cap heaps
    keep, each entry a negated hash first so the largest is the smallest hash."""

    __slots__ = ("cap", "entries")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.entries: list = []

    def push(self, entry) -> None:
        if self.cap <= 0:
            return
        if len(self.entries) < self.cap:
            heapq.heappush(self.entries, entry)
        elif entry > self.entries[0]:
            heapq.heappushpop(self.entries, entry)

    def merge(self, other: _BoundedHeap) -> None:
        """Absorb another heap's entries, keeping the cap largest of the union.

        Refused across caps: a smaller cap has already lost entries for good.
        """
        if self.cap != other.cap:
            raise ValueError(
                f"cannot merge heaps with mismatched caps: "
                f"self.cap={self.cap}, other.cap={other.cap}"
            )
        for entry in other.entries:
            self.push(entry)


class CapHeap:
    """The ``cap`` smallest record hashes seen.

    Bounded because no quota a text is given exceeds the cap; entries are bare
    ints because the survey holds one heap per text slot.
    """

    __slots__ = ("_heap",)

    def __init__(self, cap: int) -> None:
        self._heap = _BoundedHeap(cap)

    @property
    def cap(self) -> int:
        return self._heap.cap

    def offer(self, value: int) -> None:
        self._heap.push(-value)

    def merge(self, other: CapHeap) -> None:
        """Absorb another heap's values, keeping the cap smallest of the union."""
        self._heap.merge(other._heap)

    def values(self) -> tuple[int, ...]:
        return tuple(-entry for entry in self._heap.entries)

    def __len__(self) -> int:
        return len(self._heap.entries)


class KeyedCapHeap[T]:
    """The ``cap`` smallest hashes seen, each carrying an item.

    Among equal hashes the larger ``tiebreak`` is kept; it must never tie, or
    two equal hashes would compare the items.
    """

    __slots__ = ("_heap",)

    def __init__(self, cap: int) -> None:
        self._heap = _BoundedHeap(cap)

    def offer(self, value: int, item: T, tiebreak: tuple) -> None:
        self._heap.push((-value, tiebreak, item))

    def merge(self, other: KeyedCapHeap[T]) -> None:
        self._heap.merge(other._heap)

    def items(self) -> list[T]:
        """The retained items, smallest hash first."""
        return [item for _, _, item in sorted(self._heap.entries, reverse=True)]

    def __len__(self) -> int:
        return len(self._heap.entries)


def class_targets(
    available: Mapping[str, int],
    mix: Mapping[str, float],
    *,
    ceiling: int = 0,
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class write targets, and the shortfall against a requested ceiling.

    The scarcest class sets the total, ``min(available[c] / share[c])``;
    filling abundant classes to capacity would write the mixture the disk
    happens to hold. ``ceiling`` (``--training-records``) lowers that total,
    and what a ceiling above it cannot meet is returned per class (FR-139).
    """
    shares = {
        name: float(share)
        for name, share in mix.items()
        if share > 0 and available.get(name, 0) > 0
    }
    if not shares:
        raise ValueError(f"no positive share with records available: mix={dict(mix)}")
    total_share = sum(shares.values())
    supported = min(
        available[name] * total_share / share for name, share in shares.items()
    )
    total = min(supported, float(ceiling)) if ceiling > 0 else supported
    targets = {
        name: min(available[name], int(total * share / total_share))
        for name, share in shares.items()
    }
    shortfall: dict[str, int] = {}
    if ceiling > 0 and ceiling > supported:
        for name, share in shares.items():
            missing = int(ceiling * share / total_share) - targets[name]
            if missing > 0:
                shortfall[name] = missing
    return targets, shortfall


#: The sampling classes whose families split further by outcome signature
#: (FR-049 step 2); the others have no per-entity outcome to sort by.
SIGNATURE_CLASSES: frozenset[SamplingClass] = frozenset({
    SamplingClass.RESOLUTION_EFFECT, SamplingClass.REWRITE, SamplingClass.COMBAT,
    SamplingClass.CONTINUOUS,
})


def outcome_signature(record) -> frozenset[tuple[str, bool]]:
    """The distinct ``(zone outcome, changed)`` pairs over a record's entities,
    from the targets the head trains on.

    A set, not a multiset, so a wipe of three and a wipe of seven share a
    signature rather than splitting one rule by board size (FR-049 step 2).
    """
    pairs = set()
    for entry in derive_targets(record).values():
        zone = entry.fields.get(HeadField.ZONE_OUTCOME)
        outcome = ZONE_OUTCOMES[zone] if zone is not None else ZoneOutcome.STAYED
        pairs.add((outcome, entry.affected))
    return frozenset(pairs)


def signature_label(signature: frozenset[tuple[str, bool]]) -> str:
    """A signature as one stable string, for a cell key and the manifest."""
    if not signature:
        return "(empty)"
    return ",".join(sorted(
        f"{zone}:{'changed' if changed else 'unchanged'}" for zone, changed in signature
    ))


# ── condition balance (FR-056) ──────────────────────────────────────────

#: The classes whose cells split by whether a turn or step condition held:
#: the outcome-signature classes, and the decision and legality classes,
#: whose signature is the condition alone (FR-056).
CONDITION_CLASSES: frozenset[SamplingClass] = SIGNATURE_CLASSES | {
    SamplingClass.PLAYABILITY_DECISION, SamplingClass.PLAYABILITY_LEGALITY,
}

#: An activation restriction's turn and step parameters, those Forge's
#: ``checkTimingRestrictions`` reads as whose turn, phase and combat
#: (``SpellAbilityRestriction``). Casting speed is not one.
_RESTRICTION_PARAMS = frozenset({
    "PlayerTurn", "OpponentTurn", "ActivationPhases", "ActivationFirstCombat",
    "ActivationAfterBlockers",
})
#: A resolving clause's turn and step parameters (``SpellAbilityCondition``).
_CLAUSE_PARAMS = frozenset({
    "ConditionPlayerTurn", "ConditionOpponentTurn", "ConditionPhases",
    "ConditionFirstCombat",
})
#: A static's turn and step parameters, and its ``Condition$`` values of that
#: kind (``StaticAbility.checkConditions``).
_STATIC_PARAMS = frozenset({"Phases", "PlayerTurn"})
_STATIC_CONDITIONS = frozenset({"PlayerTurn", "NotPlayerTurn", "ExtraTurn"})
#: The checks a candidate fails when its turn or step restriction does not hold.
_TIMING_CHECKS = frozenset({
    PlayabilityCheck.WHOSE_TURN, PlayabilityCheck.PHASE, PlayabilityCheck.COMBAT,
})


class ConditionSide(StrEnum):
    MET = "met"
    UNMET = "unmet"


@dataclass(frozen=True, slots=True)
class TurnStepCondition:
    """A turn or step condition a record's outcome depends on: the line
    carrying it, and whether it held."""

    carrier: ProvenanceKey
    side: ConditionSide

    @property
    def label(self) -> str:
        """The condition's part of an outcome signature."""
        return f"condition:{self.side}"


def turn_step_condition(
    record: EffectRecord, lines: SidecarLookup | None,
) -> TurnStepCondition | None:
    """The turn or step condition of a record's acting line, candidate or
    listed static, and whether it held (FR-056). None where none carries one,
    or the record cannot say whether it held.

    Read from ``conditions_met`` on a resolution's effect half, from
    ``failed_checks`` on a decision's candidate, and from the line's
    ``active`` on a listed static: the acting static of a ``continuous``
    record, or a static a decision or legality record names as responsible.
    The first that carries one decides.
    """
    if lines is None:
        return None
    payload = record.payload
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        return _clause_condition(record, payload, lines)
    if record.kind is RecordKind.CONTINUOUS:
        return _static_condition(record, record.ability or (), lines)
    if isinstance(payload, PlayabilityDecisionPayload) and payload.candidates:
        candidate = payload.candidates[0]
        found = _candidate_condition(candidate, lines)
        if found is not None:
            return found
        listed = [*candidate.responsible_static, *(
            key for check in candidate.failed_checks or ()
            for key in check.responsible_static
        )]
        return _static_condition(record, listed, lines)
    if isinstance(payload, PlayabilityAttackersPayload | PlayabilityBlockersPayload):
        listed = [key for entry in payload.forbidden for key in entry.responsible_static]
        return _static_condition(record, listed, lines)
    return None


def _clause_condition(
    record: EffectRecord, payload, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """The first resolved clause whose segment carries a turn or step condition.

    A clause is named as its events' ``attributed_to`` is: ``root``, or the
    chain index counted from the first sub-ability, which is the segment
    after the root.
    """
    if not isinstance(payload, ResolutionPayload) or not payload.conditions_met:
        return None
    line = lines.first_line(record.ability or ())
    if line is None or not line.script_text:
        return None
    chain = segments(line.script_text)
    for entry in payload.conditions_met:
        index = 0 if entry.segment == "root" else _chain_index(entry.segment)
        if index is None or index >= len(chain):
            continue
        if _CLAUSE_PARAMS & params(chain[index]).keys():
            return TurnStepCondition(record.ability[0], _side(entry.met))
    return None


def _chain_index(segment: str) -> int | None:
    return int(segment) + 1 if segment.isdigit() else None


def _candidate_condition(
    candidate: Candidate, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """The candidate's own restriction, held unless it failed a timing check."""
    if candidate.failed_checks is None or not candidate.ability:
        return None
    line = lines.first_line(candidate.ability)
    if line is None or not line.script_text:
        return None
    if not _RESTRICTION_PARAMS & params(root(line.script_text)).keys():
        return None
    failed = {check.check for check in candidate.failed_checks}
    return TurnStepCondition(candidate.ability[0], _side(not failed & _TIMING_CHECKS))


def _static_condition(
    record: EffectRecord, keys: Iterable[ProvenanceKey], lines: SidecarLookup,
) -> TurnStepCondition | None:
    for key in keys:
        line = lines.line_for(key)
        if line is None or not line.script_text:
            continue
        found = params(root(line.script_text))
        if not (_STATIC_PARAMS & found.keys() or found.get("Condition") in _STATIC_CONDITIONS):
            continue
        active = _line_active(record, key)
        if active is not None:
            return TurnStepCondition(key, _side(active))
    return None


def _line_active(record: EffectRecord, key: ProvenanceKey) -> bool | None:
    """Forge's answer to whether the static ``key`` holds, read off the snapshot."""
    for entity in record.state.entities:
        for line in entity.lines or ():
            if key in line.keys and line.active is not None:
                return line.active
    return None


def _side(met: bool) -> ConditionSide:
    return ConditionSide.MET if met else ConditionSide.UNMET


def cell_signature(
    record: EffectRecord, klass: SamplingClass, lines: SidecarLookup | None,
) -> str:
    """The signature part of a record's cell: the outcome signature on the
    classes FR-049 step 2 splits, then whether a turn or step condition held
    (FR-056), so the met and unmet cells are siblings the allocator gives
    equal budgets, a short one repeating up to the reuse cap.

    A record no condition applies to keeps gen-2's label exactly.
    """
    parts = []
    if klass in SIGNATURE_CLASSES:
        parts.append(signature_label(outcome_signature(record)))
    if klass in CONDITION_CLASSES:
        condition = turn_step_condition(record, lines)
        if condition is not None:
            parts.append(condition.label)
    return "|".join(parts)


def condition_key(
    klass: SamplingClass, condition: TurnStepCondition,
) -> tuple[str, str, str]:
    """``(class, carrier, side)``: what the survey counts a condition under."""
    return (str(klass), condition.carrier.label, str(condition.side))


def one_sided_conditions(
    counts: Mapping[tuple[str, str, str], int],
) -> dict[str, list[str]]:
    """``class -> carriers`` of every condition seen on one side only, which
    no balance can even out; the manifest names them (FR-056).

    ``counts`` is keyed by :func:`condition_key`.
    """
    sides: dict[tuple[str, str], set[str]] = {}
    for (klass, carrier, side), n in counts.items():
        if n > 0:
            sides.setdefault((klass, carrier), set()).add(side)
    out: dict[str, list[str]] = {}
    for (klass, carrier), seen in sorted(sides.items()):
        if len(seen) == 1:
            out.setdefault(klass, []).append(carrier)
    return out


def text_capacity(distinct: int, *, text_cap: int, reuse_cap: int) -> int:
    """How many written records one text can supply to one cell (FR-049):
    ``min(reuse_cap × n, text_cap)``, repeats included; ``text_cap <= 0`` is
    no cap."""
    repeated = max(0, reuse_cap) * distinct
    return repeated if text_cap <= 0 else min(repeated, text_cap)


@dataclass(frozen=True, slots=True)
class CopyPlan:
    """``base`` copies, and one more when a record's hash is at most
    ``threshold``: two numbers, so a write worker needs no record list."""

    base: int = 0
    threshold: int | None = None

    def copies(self, value: int) -> int:
        extra = 1 if self.threshold is not None and value <= self.threshold else 0
        return self.base + extra


def copy_plan(smallest: Sequence[int], distinct: int, quota: int) -> CopyPlan:
    """The plan that writes ``quota`` records of a text with ``distinct`` records.

    ``smallest`` holds at least ``min(distinct, quota)`` of the text's smallest
    hashes in the cell. A quota up to ``distinct`` takes the smallest hashes
    once each, a random subset under ``--seed`` (FR-049 step 4); a larger one
    spreads evenly, so repeats differ by at most one.
    """
    ordered = sorted(smallest)
    if quota <= 0 or distinct <= 0:
        return CopyPlan()
    if quota <= distinct:
        return CopyPlan(base=0, threshold=ordered[quota - 1])
    base, extra = divmod(quota, distinct)
    return CopyPlan(base=base, threshold=ordered[extra - 1] if extra else None)


def rate_plan(distinct: int, quota: int) -> CopyPlan:
    """A copy plan for records with no ability text (combat, legality): the
    quota's rate as a hash threshold, exact in expectation rather than count."""
    if quota <= 0 or distinct <= 0:
        return CopyPlan()
    base, extra = divmod(quota, distinct)
    if not extra:
        return CopyPlan(base=base)
    return CopyPlan(base=base, threshold=int(extra / distinct * 2**64) - 1)


def game_hash(game_id: str) -> int:
    """A stable 32-bit id for a game; not ``hash()``, which is salted per
    process (FR-088a)."""
    return id_hash(game_id)


def game_disjoint_draw(game_id: str) -> float:
    """A game's position in [0, 1), from its id alone (FR-044)."""
    return game_hash(game_id) % 10**6 / 10**6
