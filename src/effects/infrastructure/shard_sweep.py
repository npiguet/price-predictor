"""Reading the validation shards in parallel, returning samples rather than shards.

The sweep that runs before training starts opens every reserved shard, and on a
curated corpus that is thousands of files and a few million records to keep four
thousand of them. Serial, it costs the better part of an hour.

Parallelizing it is only worth doing because **almost nothing a shard holds has
to come back**. Which stratum a record belongs to is decidable from its own
shard — a game never spans two — so a worker can route, sample and discard
locally, and return a digest: the game ids it saw, its class histogram, and the
few records the mixture actually keeps. Handing whole shards back instead would
spend more time pickling records than it saved reading them.

Processes rather than threads. The cost here is ``json.loads`` and building the
record objects, both of which hold the GIL; twelve threads over twelve shards
measured slightly *slower* than one thread, while twelve processes divide it.
Each worker takes a run of shards rather than one, so a process start is paid
once per run instead of once per shard, and the constants that shape the sample
are sent once per process by the initializer rather than with every task.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from effects.application.train_effect_model import CorpusSplit, HeldOutCards
    from effects.domain.records import EffectRecord

#: The strata a record can be routed to, in the order digests report them.
STRATA = ("card-disjoint", "game-disjoint")


@dataclass
class ShardDigest:
    """What one worker returns for its run of shards.

    Everything here is either small (counts, game ids) or already capped (the
    sample, the probe). A digest never carries a record the caller will discard.
    """

    #: ``(shard name, record count, game count)`` per shard, for the progress log.
    shards: list[tuple[str, int, int]] = field(default_factory=list)
    #: Games routed to each stratum, which is what a derived split accumulates.
    games: dict[str, set[str]] = field(
        default_factory=lambda: {name: set() for name in STRATA}
    )
    #: The corpus's class histogram, summed over the run's shards.
    classes: Counter[str] = field(default_factory=Counter)
    #: stratum -> class -> the records this worker kept, capped per class.
    sample: dict[str, dict[str, list]] = field(
        default_factory=lambda: {name: defaultdict(list) for name in STRATA}
    )
    #: A few records of any kind, for measuring feature widths.
    probe: list = field(default_factory=list)

    def merge(self, other: ShardDigest) -> None:
        """Fold another digest in, without re-capping the sample."""
        self.shards.extend(other.shards)
        self.classes.update(other.classes)
        for stratum in STRATA:
            self.games[stratum] |= other.games[stratum]
            for name, records in other.sample[stratum].items():
                self.sample[stratum][name].extend(records)
        self.probe.extend(other.probe)


#: Set once per worker process by the initializer. Module-level because that is
#: the only state a spawned process carries across tasks.
_HELD_OUT: HeldOutCards | None = None
_SPLIT: CorpusSplit | None = None
_QUOTA: Mapping[str, int] = {}
_PROBE_CAP = 0


def _init_worker(held_out, split, quota, probe_cap) -> None:
    global _HELD_OUT, _SPLIT, _QUOTA, _PROBE_CAP
    _HELD_OUT, _SPLIT, _QUOTA, _PROBE_CAP = held_out, split, quota, probe_cap


def digest_shards(paths: Sequence[Path]) -> ShardDigest:
    """Route, sample and discard one run of shards.

    Runs in a worker process. The split is used as given when the caller
    inherited one from a manifest; otherwise a shard's own games are routed by
    whether they name a held-out card, which is decidable here because a game
    never spans two shards.
    """
    from effects.application.train_effect_model import (
        load_shard,
        sampling_class,
        shard_games,
    )

    digest = ShardDigest()
    for path in paths:
        records = load_shard(path)
        digest.shards.append(
            (path.name, len(records), len({r.game_id for r in records})),
        )
        if _SPLIT is not None:
            card, game = _SPLIT.card_disjoint_games, _SPLIT.game_disjoint_games
        else:
            # A reserved shard's tainted games are the card-disjoint stratum and
            # its clean ones the game-disjoint stratum (FR-125).
            card, game = shard_games(records, _HELD_OUT)
        digest.games["card-disjoint"] |= {
            r.game_id for r in records if r.game_id in card
        }
        digest.games["game-disjoint"] |= {
            r.game_id for r in records if r.game_id in game
        }
        for record in records:
            digest.classes[sampling_class(record)] += 1
            if len(digest.probe) < _PROBE_CAP:
                digest.probe.append(record)
            if record.game_id in card:
                stratum = "card-disjoint"
            elif record.game_id in game:
                stratum = "game-disjoint"
            else:
                continue
            name = sampling_class(record)
            bucket = digest.sample[stratum][name]
            if len(bucket) < _QUOTA.get(name, 0):
                bucket.append(record)
        del records
    return digest


def chunk(shards: Sequence[Path], *, workers: int) -> list[list[Path]]:
    """Split the shard list into one run per worker, a few runs deep.

    Several runs per worker rather than exactly one, so a worker that draws a
    run of large shards does not hold up the sweep, and so a caller that stops
    early has a finer grain to stop on.
    """
    if not shards:
        return []
    runs = max(1, min(len(shards), workers * 4))
    size = math.ceil(len(shards) / runs)
    return [list(shards[i:i + size]) for i in range(0, len(shards), size)]


def sweep(
    shards: Sequence[Path],
    *,
    held_out,
    split,
    quota: Mapping[str, int],
    probe_cap: int,
    workers: int = 0,
) -> Iterator[ShardDigest]:
    """Yield a digest per run of shards, as the runs finish.

    Yielded rather than returned so the caller can stop once its sample is
    full; abandoning the iterator cancels the runs that have not started.

    ``quota`` caps each worker's own sample. It is the caller's per-class quota
    halved, because the union of a few runs' samples is what fills the caller's
    and a full quota from every run would hand back several times the records
    anyone keeps.
    """
    resolved = workers or (os.cpu_count() or 1)
    runs = chunk(shards, workers=resolved)
    if not runs:
        return
    share = {name: max(1, math.ceil(size / 2)) for name, size in quota.items()}
    if resolved <= 1 or len(runs) == 1:
        # No pool: one worker means this process, and one run would pay a
        # process start to read what this process is about to wait on anyway.
        # ``--workers 1`` is also the only way to sweep without subprocesses,
        # which is what a test that substitutes the shard reader needs.
        _init_worker(held_out, split, share, probe_cap)
        for run in runs:
            yield digest_shards(run)
        return
    executor = ProcessPoolExecutor(
        max_workers=min(workers or (os.cpu_count() or 1), len(runs)),
        initializer=_init_worker,
        initargs=(held_out, split, share, probe_cap),
    )
    try:
        futures = [executor.submit(digest_shards, run) for run in runs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
