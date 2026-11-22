"""Which records a curated training corpus keeps, and how many times.

Every decision is shaped so a write worker applies it to one shard alone, by
comparing a record's hash with a :class:`CopyPlan` threshold; all of it is a
pure function of the corpus, the flags and the seed (FR-052). A text over its
cap contributes a random subset rather than its first N, which would be its
earliest games' opening boards over and over.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from effects.domain.corpus_types import RecordHash
from effects.domain.seeded_draw import id_hash, keyed_hash


def record_hash(record_id: str, *, seed: int) -> RecordHash:
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

    def copy(self) -> CapHeap:
        """A heap of the same values that offers to this one never reach."""
        out = CapHeap(self.cap)
        out._heap.entries = list(self._heap.entries)
        return out

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


def validation_multipliers(
    counts: Mapping[str, int], mix: Mapping[str, float], *, reuse_cap: int,
) -> dict[str, float]:
    """The ``class_weight`` of each class of one validation stratum, which
    weighs the stratum into the training stratum's class mix (FR-059a).

    No validation record is dropped and none is repeated for the mix: the
    class most over-represented against its share weighs 1, and every other
    class weighs toward its share, at most ``reuse_cap`` a record, the most
    copies training may write of one; a class still short stays short.
    Shares are over the classes the stratum holds. A class the mix omits
    weighs 1.
    """
    held = {name: n for name, n in counts.items() if n > 0}
    shares = {name: float(mix.get(name, 0.0)) for name in held}
    shares = {name: share for name, share in shares.items() if share > 0}
    multipliers = {name: 1.0 for name in held}
    if not shares:
        return multipliers
    total_share = sum(shares.values())
    size = max(held[name] * total_share / share for name, share in shares.items())
    for name, share in shares.items():
        wanted = size * share / total_share / held[name]
        multipliers[name] = min(float(max(1, reuse_cap)), max(1.0, wanted))
    return multipliers


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
