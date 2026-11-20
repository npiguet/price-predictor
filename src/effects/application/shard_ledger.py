"""The one shard ledger every collector counts through, and the one pass that
reads a shard's records.

A collector counts its shards again after every round. The ledger keeps which
shards it has read, so each is counted once across the whole run, every
damaged shard's message, and how many games the counts left out as cut (no
game-end marker). The shard pass reads one shard's records into a per-record
visitor, notes which games ended, and stops at a damaged member or line,
keeping what came before it.

A shard is final once counted: a round's workers have all ended before
``play_round`` returns, and a shard's name carries its JVM lifetime, so no
later round reopens it. A worker the kill could not end is the exception: the
supervisor names it, and its shard is counted as it stood when read, its game
in progress cut.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from effects.application.parallel_map import run_parallel
from effects.domain.game_ends import GameEnds

if TYPE_CHECKING:
    from effects.domain.records import EffectRecord


@dataclass(frozen=True)
class ShardRead:
    """How one shard's read ended: the games no marker ended (cut, and those
    a damage hid), and the damage that ended it early, as its read's message
    (none for a whole shard)."""

    cut: frozenset[str]
    damaged: tuple[str, ...] = ()


def read_records(
    records: Iterable[EffectRecord], visit: Callable[[EffectRecord], object],
) -> ShardRead:
    """Hand ``visit`` every record ``records`` yields, up to a damaged member
    or line, noting each game's end. A record of an older generation still
    raises, as it stops every reader: it is no damage to count past."""
    from effects.infrastructure.shard_io import SHARD_DAMAGE

    ends = GameEnds()
    try:
        for record in records:
            ends.note(record)
            visit(record)
    except SHARD_DAMAGE as exc:
        return ShardRead(ends.cut, (str(exc),))
    return ShardRead(ends.cut)


class CountedShard(Protocol):
    """What the ledger reads of one shard's count, whatever else it holds."""

    records: int
    cut_games: int
    damaged: Sequence[str]


@dataclass(frozen=True)
class ShardPass:
    """How a ledger reads its fresh shards: ``count`` over each, in
    ``workers`` processes (the run's ``--workers``, never the CPU count), a
    progress line every ``progress_every`` shards. With ``context``, each
    process builds it once and ``count`` takes it first (``run_parallel``).
    ``count`` must pickle."""

    count: Callable[..., Any]
    workers: int | None
    progress_every: int
    label: str = "Counted"
    context: Callable[[], Any] | None = None


class ShardLedger:
    """The shards a run has counted, each once, and what their counts left
    out: every damaged shard's message (counted up to the damage, once, like
    any other shard) and the cut games, over the whole run and in the last
    count."""

    def __init__(self) -> None:
        self._counted: set[Path] = set()
        #: Records per shard counted so far.
        self._records: dict[Path, int] = {}
        #: Every damaged shard read so far, as its read's message.
        self.damaged: list[str] = []
        #: Gen-3 games without a game-end marker the counts left out: over
        #: the whole run, and in the last ``count``.
        self.cut_games = 0
        self.last_cut = 0

    def fresh(self, records_dir: Path, prefixes: tuple[str, ...] | None = None) -> list[Path]:
        """The shards under ``records_dir`` not counted yet, of the runs
        ``prefixes`` name (``{run_id}.``) when given."""
        from effects.infrastructure.shard_io import iter_shards

        return [
            shard for shard in iter_shards(Path(records_dir))
            if _named(shard, prefixes) and shard not in self._counted
        ]

    def count[P: CountedShard](
        self, shards: list[Path], how: ShardPass, take: Callable[[P], object],
    ) -> None:
        """Count ``shards`` as ``how`` says and hand each shard's count to
        ``take``, in shard order, noting its records, cut games and damage."""
        self.last_cut = 0
        for shard, part in zip(shards, run_parallel(
            shards, how.count, workers=how.workers, label=how.label,
            progress_every=how.progress_every, context=how.context,
        ), strict=True):
            take(part)
            self._records[shard] = part.records
            self.last_cut += part.cut_games
            self.damaged += part.damaged
        self._counted.update(shards)
        self.cut_games += self.last_cut

    def records_of(self, run_id: str) -> int:
        """Records in the counted shards of run ``run_id`` (``{run_id}.*``)."""
        prefix = f"{run_id}."
        return sum(n for shard, n in self._records.items() if shard.name.startswith(prefix))


def _named(shard: Path, prefixes: tuple[str, ...] | None) -> bool:
    return prefixes is None or shard.name.startswith(prefixes)
