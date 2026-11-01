"""Stable draws over string ids: one definition for every seeded choice.

Curation, the validation samples, placement, halves and the selection-set
freeze all choose records or games by a hash of their id, never by Python's
``hash()`` (salted per process, so a survey and a write pass would choose
differently) and never by a stateful generator (whose draw depends on the
order ids arrive in). Two hashes exist and both stay, because corpora and
probe sets already on disk were drawn through each and a rebuild has to keep
choosing what they chose:

- :func:`keyed_hash`, blake2b keyed by a seed — a family of independent draws,
  one per seed; ``record_hash`` and the validation samples draw through it.
- :func:`id_hash`, crc32 of the id alone — one fixed order; game placement and
  the probe-set selections follow it.

:func:`stable_order` and :func:`smallest_k` are the orderings new code takes,
over the keyed hash.

Torch-free.
"""

from __future__ import annotations

import hashlib
import heapq
import zlib
from collections.abc import Iterable

_HASH_BYTES = 8


def keyed_hash(value: str, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for ``value`` under ``seed``."""
    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=_HASH_BYTES,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


def id_hash(value: str) -> int:
    """A stable 32-bit draw for ``value`` alone, the same under every seed."""
    return zlib.crc32(value.encode("utf-8"))


def stable_order(ids: Iterable[str], seed: int) -> list[str]:
    """``ids`` sorted by their keyed hash under ``seed``, ties by the id itself.

    A function of each id and the seed alone, so an id keeps its place relative
    to every other id whatever else is added.
    """
    return sorted(set(ids), key=lambda value: (keyed_hash(value, seed), value))


def smallest_k(ids: Iterable[str], k: int, seed: int) -> list[str]:
    """The ``k`` ids with the smallest keyed hash under ``seed``, in that order.

    Equal to ``stable_order(ids, seed)[:k]`` without sorting everything.
    """
    if k <= 0:
        return []
    return heapq.nsmallest(
        k, set(ids), key=lambda value: (keyed_hash(value, seed), value))
