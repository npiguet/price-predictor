"""The keyword resolver a compiled matcher keeps across the records it reads.

A :class:`KeywordResolver` memoizes each provenance key's keyword off one
sidecar lookup, so one resolver serves every record read with that lookup;
a matcher holding a :class:`KeywordReading` builds it once per lookup rather
than once per record.
"""

from __future__ import annotations

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.provenance import SidecarLookup

_UNREAD = object()


class KeywordReading:
    """The resolver of the sidecar lookup last read, rebuilt for another one."""

    __slots__ = ("_lines", "_resolver")

    def __init__(self) -> None:
        self._lines: object = _UNREAD
        self._resolver: KeywordResolver | None = None

    def of(self, lines: SidecarLookup | None) -> KeywordResolver:
        """The resolver reading ``lines``."""
        if self._resolver is None or lines is not self._lines:
            self._lines, self._resolver = lines, KeywordResolver(lines)
        return self._resolver
