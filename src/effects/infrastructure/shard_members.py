"""The gzip members of a shard file, walked as decompressed chunks.

A shard is gzip members written one after another: a collector's worker
appends one per block, so a worker killed mid-write cuts only the last one.
:func:`gzip_member_chunks` hands the decompressed bytes on and knows nothing
of lines; ``shard_io`` splits them and decides what a damaged member means.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from pathlib import Path

#: Compressed bytes read at a time.
_GZIP_CHUNK = 1 << 20
#: A gzip member's first bytes: its magic and the deflate method.
_GZIP_MAGIC = b"\x1f\x8b\x08"
#: Compressed bytes tried at each place a member may start.
_MEMBER_PROBE = 1 << 16


class MemberDamage(Exception):
    """A gzip member that does not decompress, or whose check sum fails.

    ``follows`` is whether a member that does decompress starts after it:
    damage before the last member, against a damaged tail.
    """

    def __init__(self, path: Path, start: int, cause: zlib.error, *, follows: bool) -> None:
        super().__init__(f"{path}: the gzip member starting at byte {start} is corrupt ({cause})")
        self.path = path
        self.start = start
        self.cause = cause
        self.follows = follows

    @property
    def where(self) -> str:
        return str(self.args[0])


def member_follows(path: Path, after: int) -> bool:
    """Whether a gzip member that decompresses starts past byte ``after``:
    every place its magic recurs is tried, since deflate data can hold the
    three bytes by chance."""
    with Path(path).open("rb") as handle:
        handle.seek(after + 1)
        rest = handle.read()
    return any(_decompresses(rest[at:at + _MEMBER_PROBE]) for at in _magic_places(rest))


def _magic_places(data: bytes) -> Iterator[int]:
    at = data.find(_GZIP_MAGIC)
    while at >= 0:
        yield at
        at = data.find(_GZIP_MAGIC, at + 1)


def _decompresses(probe_bytes: bytes) -> bool:
    """Whether these bytes start a gzip member that yields data or ends whole."""
    probe = zlib.decompressobj(31)
    try:
        out = probe.decompress(probe_bytes)
    except zlib.error:
        return False
    return bool(out) or probe.eof


class _MemberWalk:
    """Where one file's walk over its members stands."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.decoder: zlib._Decompress | None = zlib.decompressobj(31)
        self.member_start = 0
        self.consumed = 0

    def chunks(self, data: bytes) -> Iterator[bytes]:
        """The decompressed bytes of ``data``, which may end one member and
        begin the next."""
        while data := self._begin(data):
            out, data = self._inflate(data)
            yield out

    def _begin(self, data: bytes) -> bytes:
        """``data`` with a new member started where the last one ended: the
        gzip module also skips the zero padding between members."""
        if self.decoder is not None:
            return data
        stripped = data.lstrip(b"\x00")
        self.consumed += len(data) - len(stripped)
        if stripped:
            self.decoder = zlib.decompressobj(31)
            self.member_start = self.consumed
        return stripped

    def _inflate(self, data: bytes) -> tuple[bytes, bytes]:
        """What the open member makes of ``data``, and the bytes after its end."""
        decoder = self.decoder
        try:
            out = decoder.decompress(data)
        except zlib.error as exc:
            follows = member_follows(self.path, self.member_start)
            raise MemberDamage(self.path, self.member_start, exc, follows=follows) from exc
        rest = decoder.unused_data if decoder.eof else b""
        self.consumed += len(data) - len(rest)
        if decoder.eof:
            self.decoder = None
        return out, rest


def gzip_member_chunks(path: Path) -> Iterator[bytes]:
    """The decompressed bytes of a gzip file of concatenated members, in order.

    A member cut short ends the stream after what it completed. A member that
    does not decompress, or whose check sum fails, raises
    :class:`MemberDamage` naming the byte it starts at.
    """
    walk = _MemberWalk(path)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_GZIP_CHUNK):
            yield from walk.chunks(chunk)
