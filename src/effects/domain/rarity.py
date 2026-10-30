"""How rare an ability text is, by the number of games (not records) it acted in.

One bucketing, so the trainer's epoch line (FR-055) and the evaluator's
per-text breakdown (FR-064) agree on the edges, and one rule for where a
text's game count comes from, so the weights and both reports agree on it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

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


def effective_games(
    records: Iterable[Any], text_of: Callable[[Any], str | None],
) -> dict[str, int]:
    """Distinct games contributing a record of each unique ability text.

    Games rather than records: one long game can produce hundreds of records
    of one ability, all on one board. ``records`` need only carry a
    ``game_id``.
    """
    games: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        text = text_of(record)
        if text is not None:
            games[text].add(record.game_id)
    return {text: len(ids) for text, ids in games.items()}


@dataclass(frozen=True, slots=True)
class TextGames:
    """Games per ability text: the corpus-wide ``table`` where it names a
    text, else the distinct games the records at hand span.

    A text the table does not name falls back to the records rather than to
    zero, which would weigh it heaviest of all. ``table`` None is no table and
    an empty one a table naming nothing: :meth:`games` reads them alike, and
    only :meth:`coverage` tells them apart.
    """

    observed: Mapping[str, int]
    table: Mapping[str, int] | None = None

    @classmethod
    def of(
        cls, records: Iterable[Any], text_of: Callable[[Any], str | None],
        table: Mapping[str, int] | None = None,
    ) -> TextGames:
        return cls(effective_games(records, text_of), table)

    def games(self, text: str) -> int:
        if self.table and text in self.table:
            return self.table[text]
        return self.observed.get(text, 0)

    def bucket(self, text: str) -> str:
        return rarity_bucket(self.games(text))

    def coverage(self) -> RarityCoverage | None:
        """How many of the records' distinct texts the table names; None
        without a table, which has nothing to report."""
        if self.table is None:
            return None
        return RarityCoverage(
            sum(1 for text in self.observed if text in self.table), len(self.observed),
        )


class RarityCoverage(NamedTuple):
    """How much of a set of records' distinct ability texts a rarity table
    names: a table matching nothing otherwise looks exactly like a healthy one."""

    named: int
    distinct: int

    @property
    def percent(self) -> float:
        """``named`` as a percentage of ``distinct``; 0 for no text at all."""
        return 100.0 * self.named / self.distinct if self.distinct else 0.0
