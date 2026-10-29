"""How rare an ability text is, by the number of games (not records) it acted in.

One bucketing, so the trainer's epoch line (FR-055) and the evaluator's
per-text breakdown (FR-064) agree on the edges.
"""

from __future__ import annotations

#: Labels in ascending order of games. Each is a closed range; the last is open.
RARITY_BUCKETS: tuple[str, ...] = ("1", "2-4", "5-19", "20+")

#: The smallest game count of each bucket, aligned with :data:`RARITY_BUCKETS`.
_LOWER_EDGES: tuple[int, ...] = (1, 2, 5, 20)


def rarity_bucket(games: int) -> str:
    """The bucket of a text seen in ``games`` distinct games; a count below
    one (a rounded effective count) belongs to the first."""
    index = 0
    for position, edge in enumerate(_LOWER_EDGES):
        if games >= edge:
            index = position
    return RARITY_BUCKETS[index]
