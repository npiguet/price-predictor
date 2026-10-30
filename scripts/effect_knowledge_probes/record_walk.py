"""The records a grid or a breakdown runs on, read shard by shard.

Every one-off probe picks its records the same way: walk the validation (or
training) shards in path order, keep what a predicate accepts, cap how many
come from one game — a game repeats its board from record to record, and forty
records of one game would read one board forty times — and stop at a count.
Reading in path order rather than sampling keeps the choice the same for every
arm on one corpus without a frozen list.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path


def qualifying(
    directories: Iterable[Path],
    choose: Callable[[object], object | None],
    count: int,
    *,
    per_game: int | None = 1,
    shards: int | None = None,
) -> list:
    """The first ``count`` non-None ``choose(record)`` values, at most ``per_game`` a game.

    ``choose`` returns what to keep for a qualifying record — the record
    itself, or the record with what made it qualify — and None for one that
    does not. ``shards`` bounds the shards read per directory; ``per_game``
    None takes every qualifying record of a game.
    """
    from effects.infrastructure.shard_io import iter_shards, read_shard

    chosen: list = []
    per: Counter[str] = Counter()
    for directory in directories:
        for shard in iter_shards(Path(directory))[:shards]:
            for record in read_shard(shard):
                if per_game is not None and per[record.game_id] >= per_game:
                    continue
                found = choose(record)
                if found is None:
                    continue
                per[record.game_id] += 1
                chosen.append(found)
                if len(chosen) >= count:
                    return chosen
    return chosen
