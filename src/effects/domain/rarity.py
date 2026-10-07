"""How rare an ability text is, by the number of games it acted in.

One bucketing serves three readers: the trainer's epoch line (FR-055), the
evaluator's per-text breakdown (FR-064) and the curated-corpus manifest's list
of held-out texts under five games (FR-053). They have to agree on the edges,
or a text the epoch line counts as rare would be reported as common beside it.

Games, not records: one long game can produce hundreds of records of one
ability, which says nothing about how often the text was seen across boards.
"""

from __future__ import annotations

#: Labels in ascending order of games. Each is a closed range; the last is open.
RARITY_BUCKETS: tuple[str, ...] = ("1", "2-4", "5-19", "20+")

#: The smallest game count of each bucket, aligned with :data:`RARITY_BUCKETS`.
_LOWER_EDGES: tuple[int, ...] = (1, 2, 5, 20)


def rarity_bucket(games: int) -> str:
    """The bucket of a text seen in ``games`` distinct games.

    A count below one belongs to the first bucket: a text with records always
    acted in at least one game, so zero reaches here only through a rounded
    effective count, and it is the rarest kind of text there is.
    """
    index = 0
    for position, edge in enumerate(_LOWER_EDGES):
        if games >= edge:
            index = position
    return RARITY_BUCKETS[index]
