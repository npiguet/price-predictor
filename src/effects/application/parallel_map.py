"""One task over many shards, in a process pool or in process, with progress."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def progress_line(*, done: int, total: int, elapsed: float) -> str:
    """``done of total`` with the rate and the time left."""
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate if rate > 0 else 0.0
    if rate <= 0:
        return f"{done} of {total} shard(s)"
    return (
        f"{done} of {total} shard(s), {rate * 60:.0f}/min, "
        f"~{left / 60:.0f} min left"
    )


def run_parallel[T, R](
    items: Sequence[T],
    task: Callable[[T], R],
    *,
    workers: int | None,
    label: str,
    progress_every: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> Iterator[R]:
    """``task`` over every item, results in input order, logging progress.

    ``workers`` defaults to the usable cores, capped at the item count; one or
    fewer runs everything in this process, otherwise ``task`` and
    ``initializer`` must pickle. Progress is logged every ``progress_every``
    items and at the end, never on a pass no longer than one interval.
    """
    total = len(items)
    started = time.monotonic()

    def report(done: int) -> None:
        if not progress_every or total <= progress_every:
            return
        if done % progress_every == 0 or done == total:
            logger.info("%s %s", label, progress_line(
                done=done, total=total, elapsed=time.monotonic() - started,
            ))

    if workers is None:
        workers = os.process_cpu_count() or 1
    if min(workers, total) <= 1:
        if initializer is not None:
            initializer(*initargs)
        for done, item in enumerate(items, start=1):
            yield task(item)
            report(done)
        return
    with ProcessPoolExecutor(
        max_workers=min(workers, total), initializer=initializer, initargs=initargs,
    ) as pool:
        for done, result in enumerate(pool.map(task, items, chunksize=1), start=1):
            yield result
            report(done)
