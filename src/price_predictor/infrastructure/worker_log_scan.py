"""Reading a run's worker logs for failure lines, a round at a time.

Each scan reads only what the logs gained since the last one, from the last
complete line on, so a line the worker is still writing is read whole next
time. A slot's log rotated past its size cap (``WorkerLogFiles``) is noticed
by its file changing identity: the rest of the rotated file is read before
the new one. Rotated twice between scans, the generation read so far is gone
(the second rotation replaced its ``.1``): the loss is counted as
``log_unread``, and the ``.1`` there now, never read, is read whole.

The same reading follows each slot's status lines, the counts each worker
JVM prints and no failure lines (``worker_status.SlotStatuses``), and keeps
the run's first one, whose engine every later lifetime's is compared with
(``worker_status.judge_engine``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from price_predictor.domain.worker_failures import WorkerFailure, WorkerFailures
from price_predictor.domain.worker_status import (
    EngineCheck,
    EngineVerdict,
    SlotStatuses,
    WorkerStatus,
    directed_fork_tally_document,
    engine_document,
    fork_refusal_document,
    hook_failure_document,
    judge_engine,
)

if TYPE_CHECKING:
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles


class WorkerLogScan:
    """Failure lines added to a set of log files since the last scan."""

    def __init__(self) -> None:
        #: Per log path: the file's identity and how far it has been read.
        self._read: dict[Path, tuple[int, int]] = {}
        #: Per log path: the status lines its worker JVMs printed.
        self._statuses: dict[Path, SlotStatuses] = {}
        #: The first status line any scan read, in path order within a scan.
        self._first: WorkerStatus | None = None

    def scan(self, paths: Iterable[Path]) -> WorkerFailures:
        """The failure lines ``paths`` gained since the last scan."""
        failures = WorkerFailures()
        for path in paths:
            self._scan_one(Path(path), failures)
        if self._first is None:
            self._first = next(
                (status for slot in self._statuses.values()
                 for status in slot.lifetimes.values()), None)
        return failures

    def first_status(self) -> WorkerStatus | None:
        """The run's first status line: its first worker lifetime's engine."""
        return self._first

    def engine_verdict(self) -> EngineVerdict:
        """The first lifetime's engine, and every lifetime read so far whose
        engine differs from it (``worker_status.judge_engine``)."""
        return judge_engine(self._first, self._statuses.values())

    def hook_failures(self) -> dict[str, dict[str, int]]:
        """The hook-failure counts every log read so far reports, summed
        (``worker_status.hook_failure_document``)."""
        return hook_failure_document(self._statuses.values())

    def fork_refusals(self) -> dict[str, int]:
        """The fork refusals every log read so far reports, summed
        (``worker_status.fork_refusal_document``)."""
        return fork_refusal_document(self._statuses.values())

    def directed_fork_tally(self) -> dict[str, dict[str, int]]:
        """The directed fork tallies every log read so far reports, summed
        (``worker_status.directed_fork_tally_document``)."""
        return directed_fork_tally_document(self._statuses.values())

    def _scan_one(self, path: Path, failures: WorkerFailures) -> None:
        statuses = self._statuses.setdefault(path, SlotStatuses())
        try:
            identity = os.stat(path).st_ino
        except OSError:
            return
        known, offset = self._read.get(path, (identity, 0))
        if known != identity:
            # Rotated: the file read so far is ``.1`` now, unless a second
            # rotation replaced it; finish whichever generation is there.
            rotated = path.with_name(path.name + ".1")
            try:
                rotated_identity = os.stat(rotated).st_ino
            except OSError:
                rotated_identity = None
            if rotated_identity == known:
                _read_lines(rotated, offset, failures, statuses)
            else:
                failures.add(
                    WorkerFailure.LOG_UNREAD,
                    f"{path}: rotated at least twice between scans; the "
                    "generation read so far is gone",
                )
                if rotated_identity is not None:
                    _read_lines(rotated, 0, failures, statuses)
            offset = 0
        self._read[path] = (identity, _read_lines(path, offset, failures, statuses))


class WorkerLogReport:
    """A worker pool's logs, scanned on demand: what each scan found new,
    and the run's totals."""

    def __init__(
        self, logs: WorkerLogFiles, worker_count: int, *, allow_degraded: bool = False,
    ) -> None:
        self._logs = logs
        self._worker_count = worker_count
        self._scan = WorkerLogScan()
        #: Every failure line found so far, by kind.
        self.totals = WorkerFailures()
        #: The judgement of the workers' engine, at every scan.
        self._engine = EngineCheck(allow_degraded=allow_degraded)

    def scan(self) -> WorkerFailures:
        """The failure lines every slot's log gained since the last scan,
        added to :attr:`totals`."""
        found = self._scan.scan(
            self._logs.path_for(slot) for slot in range(self._worker_count)
        )
        self.totals.merge(found)
        return found

    def review_engine(self) -> str | None:
        """Judge the engine the status lines read so far show; the
        ``--allow-degraded`` warning to give now, at most once a run. A reason
        to stop lands in :attr:`engine_refusal` and stays."""
        return self._engine.review(self._scan.engine_verdict())

    @property
    def engine_refusal(self) -> str | None:
        """Why the run must stop for its workers' engine, or None."""
        return self._engine.refusal

    def worker_engine(self) -> dict | None:
        """The first worker lifetime's engine as ``run.json`` records it, or
        None before any worker printed its status line."""
        first = self._scan.first_status()
        return None if first is None else engine_document(first)

    def hook_failures(self) -> dict[str, dict[str, int]]:
        """The latest hook-failure counts the workers printed, summed over
        every worker JVM of the run."""
        return self._scan.hook_failures()

    def fork_refusals(self) -> dict[str, int]:
        """The latest fork refusal tallies the workers printed, summed over
        every worker JVM of the run: expected, never a failure."""
        return self._scan.fork_refusals()

    def directed_fork_tally(self) -> dict[str, dict[str, int]]:
        """The latest directed fork tallies the workers printed, per rule,
        summed over every worker JVM of the run."""
        return self._scan.directed_fork_tally()

    def manifest_fields(self) -> dict:
        """The totals as ``run.json`` records them."""
        return {
            "worker_failures": self.totals.as_dict(),
            "worker_failure_examples": dict(self.totals.examples),
            "hook_failures": self.hook_failures(),
            "fork_refusals": self.fork_refusals(),
            "directed_fork_tally": self.directed_fork_tally(),
            "worker_engine": self.worker_engine(),
        }


def _read_lines(
    path: Path, offset: int, failures: WorkerFailures, statuses: SlotStatuses,
) -> int:
    """Count the failure lines of ``path`` from ``offset`` to its last complete
    line, and follow its status lines; the offset just past that line."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
    except OSError:
        return offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return offset
    for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
        failures.add_line(line)
        statuses.add_line(line)
    return offset + end + 1
