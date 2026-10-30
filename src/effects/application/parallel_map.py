"""One task over many shards, in a process pool or in process, with progress."""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

#: The context this pool worker process built, once, at its start.
_CONTEXT: Any = None


def _install_context(factory: Callable[[], Any]) -> None:
    global _CONTEXT
    _CONTEXT = factory()


def _call_in_context(task: Callable[[Any, Any], Any], item: Any) -> Any:
    return task(_CONTEXT, item)


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
    task: Callable[..., R],
    *,
    workers: int | None,
    label: str,
    progress_every: int,
    context: Callable[[], Any] | None = None,
) -> Iterator[R]:
    """``task`` over every item, results in input order, logging progress.

    With ``context``, each process calls it once and every item is run as
    ``task(built, item)``, so what a task reads per process, such as a sidecar
    cache, is built once per process rather than once per item; without it,
    ``task(item)``. ``workers`` defaults to the usable cores, capped at the
    item count; one or fewer runs everything in this process, otherwise
    ``task`` and ``context`` must pickle. Progress is logged every
    ``progress_every`` items and at the end, never on a pass no longer than
    one interval.
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
        run = task if context is None else functools.partial(task, context())
        for done, item in enumerate(items, start=1):
            yield run(item)
            report(done)
        return
    with ProcessPoolExecutor(
        max_workers=min(workers, total),
        initializer=None if context is None else _install_context,
        initargs=() if context is None else (context,),
    ) as pool:
        run = task if context is None else functools.partial(_call_in_context, task)
        for done, result in enumerate(pool.map(run, items, chunksize=1), start=1):
            yield result
            report(done)
