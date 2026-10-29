"""Which records a curated training corpus keeps, and how many times.

Every decision is shaped so a write worker applies it to one shard alone, by
comparing a record's hash with a :class:`CopyPlan` threshold; all of it is a
pure function of the corpus, the flags and the seed (FR-052). A text over its
cap contributes a random subset rather than its first N, which would be its
earliest games' opening boards over and over.
"""

from __future__ import annotations

import hashlib
import heapq
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
from effects.domain.effect_targets import derive_targets
from effects.domain.sampling import SamplingClass

_HASH_BYTES = 8


def record_hash(record_id: str, *, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for one record.

    Not the built-in ``hash()``, which is salted per process, so the survey and
    write passes would silently admit different records (FR-088a).
    """
    digest = hashlib.blake2b(
        record_id.encode("utf-8"),
        digest_size=_HASH_BYTES,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


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
    return zlib.crc32(game_id.encode("utf-8"))


def game_disjoint_draw(game_id: str) -> float:
    """A game's position in [0, 1), from its id alone (FR-044)."""
    return game_hash(game_id) % 10**6 / 10**6


def in_game_disjoint_stratum(
    game_id: str,
    has_keyword_combat: bool,
    share: float,
    keyword_share: float,
) -> bool:
    """Whether a game not held out enters the game-disjoint stratum (FR-044–046).

    A pure function of the game, so a rebuild over a grown corpus places every
    game it placed before (SC-007). A game holding a qualifying keyword combat
    takes the higher threshold, since those are what gate 2 scores.
    """
    threshold = keyword_share if has_keyword_combat else share
    return game_disjoint_draw(game_id) < threshold
