"""The records a grid or a breakdown runs on, read shard by shard.

Every one-off probe picks its records the same way: walk the validation (or
training) shards the manifest lists (``ProbeContext.stratum_shards``), in order,
keep what a predicate accepts, cap how many come from one game — a game repeats
its board from record to record, and forty records of one game would read one
board forty times — and stop at a count.
Reading in path order rather than sampling keeps the choice the same for every
arm on one corpus without a frozen list.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypedDict, Unpack


def _first_seen(rows: Iterable, seen: set[str]) -> Iterator:
    """``rows`` less those whose record id ``seen`` already holds, adding the rest to it."""
    for record in rows:
        if record.record_id not in seen:
            seen.add(record.record_id)
            yield record


def records(shards: Iterable[Path], *, distinct: bool = True) -> Iterator:
    """Every record of the listed ``shards``, in the order given.

    Gen-3 shards only. A curated stratum repeats a record's row for each copy
    curation wrote of it; ``distinct`` (the default) yields each record once,
    as every probe reads it (FR-070), and False every row, as training does.
    """
    from effects.infrastructure.shard_io import read_shard, require_generation

    seen: set[str] = set()
    chosen = list(shards)
    require_generation(chosen)
    for shard in chosen:
        rows = read_shard(shard)
        yield from _first_seen(rows, seen) if distinct else rows


class WalkLimits(TypedDict, total=False):
    """The keyword limits of ``qualifying``."""

    #: At most this many qualifying records from one game; None takes every one.
    per_game: int | None


def qualifying(
    shards: Iterable[Path],
    choose: Callable[[object], object | None],
    count: int,
    **limits: Unpack[WalkLimits],
) -> list:
    """The first ``count`` non-None ``choose(record)`` values, at most ``per_game`` a game.

    ``choose`` returns what to keep for a qualifying record — the record
    itself, or the record with what made it qualify — and None for one that
    does not. ``per_game``
    (1 by default) None takes every qualifying record of a game.
    """
    per_game = limits.get("per_game", 1)
    chosen: list = []
    per: Counter[str] = Counter()
    for record in records(shards):
        if per_game is not None and per[record.game_id] >= per_game:
            continue
        found = choose(record)
        if found is None:
            continue
        per[record.game_id] += 1
        chosen.append(found)
        if len(chosen) >= count:
            break
    return chosen


def with_record(find: Callable[[object], tuple | None]) -> Callable[[object], tuple | None]:
    """A ``choose`` keeping ``(record, *find(record))`` wherever ``find`` finds something."""

    def choose(record):
        found = find(record)
        return None if found is None else (record, *found)

    return choose
