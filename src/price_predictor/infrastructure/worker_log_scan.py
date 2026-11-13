"""Reading a run's worker logs for failure lines, a round at a time.

Each scan reads only what the logs gained since the last one, from the last
complete line on, so a line the worker is still writing is read whole next
time. A slot's log rotated past its size cap (``WorkerLogFiles``) is noticed
by its file changing identity: the rest of the rotated file is read before
the new one. Rotated twice between scans, the generation read so far is gone
(the second rotation replaced its ``.1``): the loss is counted as
``log_unread``, and the ``.1`` there now, never read, is read whole.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from price_predictor.domain.worker_failures import WorkerFailure, WorkerFailures

if TYPE_CHECKING:
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles


class WorkerLogScan:
    """Failure lines added to a set of log files since the last scan."""

    def __init__(self) -> None:
        #: Per log path: the file's identity and how far it has been read.
        self._read: dict[Path, tuple[int, int]] = {}

    def scan(self, paths: Iterable[Path]) -> WorkerFailures:
        """The failure lines ``paths`` gained since the last scan."""
        failures = WorkerFailures()
        for path in paths:
            self._scan_one(Path(path), failures)
        return failures

    def _scan_one(self, path: Path, failures: WorkerFailures) -> None:
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
                _read_lines(rotated, offset, failures)
            else:
                failures.add(
                    WorkerFailure.LOG_UNREAD,
                    f"{path}: rotated at least twice between scans; the "
                    "generation read so far is gone",
                )
                if rotated_identity is not None:
                    _read_lines(rotated, 0, failures)
            offset = 0
        self._read[path] = (identity, _read_lines(path, offset, failures))


class WorkerLogReport:
    """A worker pool's logs, scanned on demand: what each scan found new,
    and the run's totals."""

    def __init__(self, logs: WorkerLogFiles, worker_count: int) -> None:
        self._logs = logs
        self._worker_count = worker_count
        self._scan = WorkerLogScan()
        #: Every failure line found so far, by kind.
        self.totals = WorkerFailures()

    def scan(self) -> WorkerFailures:
        """The failure lines every slot's log gained since the last scan,
        added to :attr:`totals`."""
        found = self._scan.scan(
            self._logs.path_for(slot) for slot in range(self._worker_count)
        )
        self.totals.merge(found)
        return found

    def manifest_fields(self) -> dict:
        """The totals as ``run.json`` records them."""
        return {
            "worker_failures": self.totals.as_dict(),
            "worker_failure_examples": dict(self.totals.examples),
        }


def _read_lines(path: Path, offset: int, failures: WorkerFailures) -> int:
    """Count the failure lines of ``path`` from ``offset`` to its last complete
    line; the offset just past that line."""
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
    return offset + end + 1
