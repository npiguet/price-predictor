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
(``worker_status.judge_engine``). A line carrying the status prefix that
does not parse is a failure line of its own kind,
``status_line_unreadable``; and a run whose workers named their shards or
played a match while no lifetime printed a line that parses is refused.
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
from price_predictor.infrastructure.records_conversion import recorded_forge_commit

if TYPE_CHECKING:
    from price_predictor.infrastructure.worker_log_files import WorkerLogFiles


class WorkerLogScan:
    """Failure lines added to a set of log files since the last scan."""

    def __init__(self) -> None:
        #: Per log path: the file's identity and how far it has been read.
        self._read: dict[Path, tuple[int, int]] = {}
        #: Per log path: the status lines its worker JVMs printed.
        self._statuses: dict[Path, SlotStatuses] = {}
        #: The first status line any scan read, in path order within a scan.
        self._first: WorkerStatus | None = None
        #: Whether a worker had named its shard by the scan before the last.
        self._named_shard_before = False

    def scan(self, paths: Iterable[Path]) -> WorkerFailures:
        """The failure lines ``paths`` gained since the last scan."""
        failures = WorkerFailures()
        self._named_shard_before = self.named_shard()
        for path in paths:
            self._scan_one(Path(path), failures)
        if self._first is None:
            self._first = next(
                (status for slot in self._statuses.values()
                 for status in slot.lifetimes.values()), None)
        return failures

    def named_shard(self, *, before_the_last_scan: bool = False) -> bool:
        """Whether a worker printed the line naming its shard, in any scan or
        in a scan before the last one: a worker prints its status line just
        after, so one seen a status interval ago without one never printed it."""
        if before_the_last_scan:
            return self._named_shard_before
        return any(slot.named_shard for slot in self._statuses.values())

    def first_status(self) -> WorkerStatus | None:
        """The run's first status line: its first worker lifetime's engine."""
        return self._first

    def engine_verdict(
        self, conversion_commit: str | None = None, *, silent: bool = False,
    ) -> EngineVerdict:
        """The first lifetime's engine against ``conversion_commit``, and
        every lifetime read so far whose engine differs from it
        (``worker_status.judge_engine``); ``silent`` when the caller saw
        workers that should have printed a status line by now."""
        return judge_engine(
            self._first, self._statuses.values(),
            conversion_commit=conversion_commit, silent=silent,
        )

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
        identity = _inode(path)
        if identity is None:
            return
        known, offset = self._read.get(path, (identity, 0))
        if known != identity:
            self._finish_rotated(path, known, offset, failures)
            offset = 0
        self._read[path] = (identity, _read_lines(path, offset, failures, statuses))

    def _finish_rotated(
        self, path: Path, known: int, offset: int, failures: WorkerFailures,
    ) -> None:
        """Rotated: the file read so far is ``.1`` now, unless a second
        rotation replaced it; finish whichever generation is there."""
        statuses = self._statuses[path]
        rotated = path.with_name(path.name + ".1")
        rotated_identity = _inode(rotated)
        if rotated_identity == known:
            _read_lines(rotated, offset, failures, statuses)
            return
        failures.add(
            WorkerFailure.LOG_UNREAD,
            f"{path}: rotated at least twice between scans; the "
            "generation read so far is gone",
        )
        if rotated_identity is not None:
            _read_lines(rotated, 0, failures, statuses)


class WorkerLogReport:
    """A worker pool's logs, scanned on demand: what each scan found new,
    and the run's totals."""

    def __init__(
        self, logs: WorkerLogFiles, worker_count: int, *, allow_degraded: bool = False,
        run_dir: Path | None = None,
    ) -> None:
        self._logs = logs
        self._worker_count = worker_count
        self._scan = WorkerLogScan()
        #: Every failure line found so far, by kind.
        self.totals = WorkerFailures()
        #: The judgement of the workers' engine, at every scan.
        self._engine = EngineCheck(allow_degraded=allow_degraded)
        #: The Forge commit of the conversion the run's ``run.json`` (in
        #: ``run_dir``) records, every lifetime's engine judged against it.
        self._conversion_commit = None if run_dir is None else recorded_forge_commit(run_dir)

    def scan(self) -> WorkerFailures:
        """The failure lines every slot's log gained since the last scan,
        added to :attr:`totals`."""
        found = self._scan.scan(
            self._logs.path_for(slot) for slot in range(self._worker_count)
        )
        self.totals.merge(found)
        return found

    def review_engine(self, *, played: bool = False, ended: bool = False) -> str | None:
        """Judge the engine the status lines read so far show; the
        ``--allow-degraded`` warning to give now, at most once a run. A reason
        to stop lands in :attr:`engine_refusal` and stays.

        No status line that parses is a reason once ``played`` (the caller
        counted a match played before the last scan), once a worker named
        its shard a scan before the last, or, ``ended`` (every worker has
        ended, so its log is whole), once one named its shard at all."""
        silent = played or self._scan.named_shard(before_the_last_scan=not ended)
        return self._engine.review(
            self._scan.engine_verdict(self._conversion_commit, silent=silent))

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


def _inode(path: Path) -> int | None:
    """The file's identity, or None when it cannot be statted."""
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


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
        if not statuses.add_line(line):
            failures.add(WorkerFailure.STATUS_LINE_UNREADABLE, line)
    return offset + end + 1
