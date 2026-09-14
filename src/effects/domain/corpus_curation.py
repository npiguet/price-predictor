"""Which records a curated training corpus keeps (FR-138, FR-139, FR-140).

Two decisions, both shaped so a worker can apply them to one shard without
talking to any other worker. The per-text cap becomes a numeric threshold on a
hash of the record id, computed once from a survey of the whole corpus; a
worker then keeps a record by comparing one number. The class mixture becomes a
per-class record target, applied the same way.

The cap keeps a *random* subset rather than the first N: a text's earliest
records are its earliest games, and keeping those would describe one opening
board over and over — the bias the mana reservoir sampler exists to avoid, on
the other side of the pipeline.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping

#: uint64, which is the width the threshold comparison is over.
_HASH_BYTES = 8


def record_hash(record_id: str, *, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for one record.

    ``blake2b`` keyed by the seed rather than the built-in ``hash()``, which is
    salted per process: a threshold computed in the survey pass would admit a
    different set of records in the write pass, and nothing would say so
    (FR-088a).
    """
    digest = hashlib.blake2b(
        record_id.encode("utf-8"),
        digest_size=_HASH_BYTES,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


class CapHeap:
    """The ``cap`` smallest record hashes seen for one ability text.

    A max-heap of negated values, so the largest retained hash sits at the root
    and is exactly the threshold that admits ``cap`` records and no more. Held
    bounded rather than collecting every hash: the corpus has tens of millions
    of records and this runs once per unique text.
    """

    __slots__ = ("cap", "_heap")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._heap: list[int] = []

    def offer(self, value: int) -> None:
        if self.cap <= 0:
            return
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, -value)
        elif -value > self._heap[0]:
            heapq.heappushpop(self._heap, -value)

    def merge(self, other: CapHeap) -> None:
        """Absorb another heap's values, keeping the cap smallest of the union.

        Valid only when both heaps share the same cap, because the k smallest of
        a union are drawn from each side's own k smallest. This precondition is
        enforced to prevent silent data loss: if the other heap discarded values
        due to a smaller cap, they are irrecoverably lost here.
        """
        if self.cap != other.cap:
            raise ValueError(
                f"cannot merge heaps with mismatched caps: "
                f"self.cap={self.cap}, other.cap={other.cap}"
            )
        for negated in other._heap:
            self.offer(-negated)

    def values(self) -> tuple[int, ...]:
        return tuple(-negated for negated in self._heap)

    def threshold(self) -> int | None:
        """The largest retained hash, or None while under the cap.

        None means every record of this text is kept, which is the case for
        every text the cap never reaches — most of them.
        """
        if self.cap <= 0 or len(self._heap) < self.cap:
            return None
        return -self._heap[0]


def keeps(value: int, threshold: int | None) -> bool:
    """Whether a record's hash is admitted by its text's threshold."""
    return True if threshold is None else value <= threshold


def class_targets(
    available: Mapping[str, int],
    mix: Mapping[str, float],
    *,
    ceiling: int = 0,
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class write targets, and the shortfall against a requested ceiling.

    The scarcest class sets the size of the whole dataset: the largest total
    the mixture can be written at is ``min(available[c] / share[c])``, and
    every class then takes its share of that. Filling the abundant classes to
    their own capacity instead would write the mixture the disk happens to
    hold, which is the shape the dataset exists to correct.

    ``ceiling`` (``--training-records``) lowers that total. A ceiling above
    what the mixture supports cannot be met, and the difference is returned per
    class rather than silently under-filled (FR-139).
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
