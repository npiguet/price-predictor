"""Reading a run's worker logs for failure lines, a round at a time.

Each scan reads only what the logs gained since the last one, from the last
complete line on, so a line the worker is still writing is read whole next
time. A slot's log rotated past its size cap (``WorkerLogFiles``) is noticed
by its file changing identity: the rest of the rotated file is read before
the new one.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from effects.domain.worker_failures import WorkerFailures


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
            # Rotated: the file read so far is now ``.1``; finish it first.
            rotated = path.with_name(path.name + ".1")
            try:
                if os.stat(rotated).st_ino == known:
                    _read_lines(rotated, offset, failures)
            except OSError:
                pass
            offset = 0
        self._read[path] = (identity, _read_lines(path, offset, failures))


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
