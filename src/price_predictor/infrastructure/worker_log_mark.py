"""Where a slot's log stood at a worker's spawn: what the worker wrote since,
whether it has printed its startup line, and its stop line."""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.worker_stops import STARTUP_LINE, STOPPED_LINE

#: The most a startup scan reads at once: Forge prints a line per custom card
#: before its startup line, megabytes with a staged variant tree, so the scan
#: reads on from where it stopped, in chunks, and holds none of the log.
_SCAN_CHUNK_BYTES = 1024 * 1024


@dataclasses.dataclass
class _LogMark:
    """Where a slot's log stood when a worker was spawned: its identity and
    size, so what that worker wrote can be read back. The mark also keeps the
    worker's startup scan: how far the log has been searched for the startup
    line (``scanned``, None before the first search), the tail of the last
    chunk (a line split across two reads) and whether the line was seen."""

    path: Path
    identity: int | None
    size: int
    scanned: int | None = None
    carry: bytes = b""
    started: bool = False

    @classmethod
    def of(cls, path: Path) -> _LogMark:
        try:
            stat = os.stat(path)
        except OSError:
            return cls(path, None, 0)
        return cls(path, stat.st_ino, stat.st_size)

    def written_since(self, *, head: int | None = None, tail: int | None = None) -> str:
        """What the log gained since the mark: its first ``head`` bytes, or
        its last ``tail``. A log rotated at the spawn is read from its start."""
        try:
            stat = os.stat(self.path)
            start = self.size if stat.st_ino == self.identity else 0
            with open(self.path, "rb") as handle:
                if tail is not None:
                    start = max(start, stat.st_size - tail)
                handle.seek(start)
                chunk = handle.read(head if head is not None else -1)
        except OSError:
            return ""
        return chunk.decode("utf-8", errors="replace")


def _past_forge_start(mark: _LogMark | None) -> bool:
    """Whether the worker whose log ``mark`` marks has finished starting
    Forge: its startup line is anywhere in what it wrote since its spawn.
    Each call reads only what the log gained since the last; once the line has
    been seen the log is not read again."""
    if mark is None:
        return False
    if not mark.started:
        _scan_for_startup(mark)
    return mark.started


def _scan_for_startup(mark: _LogMark) -> None:
    """Search the log's new bytes for the startup line."""
    try:
        size = os.stat(mark.path).st_size
        if mark.scanned is None or size < mark.scanned:
            mark.scanned = mark.size if _same_file(mark) else 0
            mark.carry = b""
        with open(mark.path, "rb") as handle:
            handle.seek(mark.scanned)
            _read_through(handle, mark)
    except OSError:
        return


def _same_file(mark: _LogMark) -> bool:
    """Whether the log is still the file the mark was taken of."""
    return os.stat(mark.path).st_ino == mark.identity


def _read_through(handle: IO[bytes], mark: _LogMark) -> None:
    """Read ``handle`` to its end a chunk at a time, advancing the mark's
    scan and setting ``started`` on the startup line. The last bytes of a
    chunk carry over, for a line split across two reads."""
    marker = STARTUP_LINE.encode("utf-8")
    while not mark.started and (chunk := handle.read(_SCAN_CHUNK_BYTES)):
        mark.scanned += len(chunk)
        window = mark.carry + chunk
        mark.started = marker in window
        mark.carry = window[-(len(marker) - 1):]


def stop_line_of(text: str) -> re.Match[str] | None:
    """The last stop line in ``text``, or None."""
    found = None
    for found in STOPPED_LINE.finditer(text):  # noqa: B007 — the last one
        pass
    return found
