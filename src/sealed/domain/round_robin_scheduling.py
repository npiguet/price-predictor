"""Scheduling helpers for round-robin deck evaluation.

These helpers arrange A-deck × B-deck matches for distribution across worker
processes. The round-robin aggregation in ``round_robin_results`` depends on
the row-major ordering produced here (match index ``k`` → A deck ``k // n_b``
vs B deck ``k % n_b``), so keep the two modules in lockstep.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def row_major_pairings(
    a_decks: list[list[str]], b_decks: list[list[str]],
) -> list[tuple[list[str], list[str]]]:
    """Return every (a, b) pair in row-major order (outer loop over a_decks)."""
    return [(a, b) for a in a_decks for b in b_decks]


def split_evenly(seq: list[T], n: int) -> list[list[T]]:
    """Split ``seq`` into ``n`` chunks as evenly as possible, preserving order.

    The first ``len(seq) % n`` chunks receive one extra element.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    total = len(seq)
    chunk_size, remainder = divmod(total, n)
    chunks: list[list[T]] = []
    offset = 0
    for i in range(n):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(seq[offset:offset + size])
        offset += size
    return chunks
