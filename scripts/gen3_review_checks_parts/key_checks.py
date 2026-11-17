"""Keys the records name against the converted sidecars, read through the effects join."""

from __future__ import annotations

from effects.domain.provenance import KeyResolution

from .model import TallyCheck, entities_of
from .sidecars import UNDESCRIBED, Sidecars

POSITIONS = ("acting-keys", "board-lines")
OUTCOMES = (*(resolution.value for resolution in KeyResolution), UNDESCRIBED)


def classify(sidecars: Sidecars, key: dict) -> str:
    """What the join answers for ``key``: a ``KeyResolution`` value (``line``,
    ``ambiguous``, ``dropped``, ``runtime-only``, ``unconverted``, ``unconfigured-tree``),
    or ``undescribed``."""
    return sidecars.resolution_of(key)


def _board_keys(record: dict) -> list[dict]:
    return [
        key
        for entity in entities_of(record)
        for line in entity.get("lines") or ()
        for key in line.get("keys") or ()
    ]


class JoinedKeys(TallyCheck):
    """Acting and board keys by the join's answer; a board key counts once per slot."""

    NAMES = tuple(f"{p}-{o}" for p in POSITIONS for o in OUTCOMES)

    def __init__(self, sidecars: Sidecars) -> None:
        super().__init__()
        self.sidecars = sidecars

    def feed(self, record: dict) -> None:
        self._tally("acting-keys", record.get("ability") or ())
        slots = [slot for key in _board_keys(record) for slot in self.sidecars.board_slots(key)]
        self._tally("board-lines", slots)

    def _tally(self, position: str, keys) -> None:
        for key in keys:
            answer = classify(self.sidecars, key)
            for outcome in OUTCOMES:
                self.add(f"{position}-{outcome}", answer == outcome)
