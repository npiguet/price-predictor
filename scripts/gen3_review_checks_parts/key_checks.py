"""Keys the records name against the converted sidecars: which resolve to one line."""

from __future__ import annotations

from .model import TallyCheck, entities_of
from .sidecars import Sidecars

POSITIONS = ("acting-keys", "board-lines")
KINDS = ("keyword", "spell", "trigger", "static", "replacement")


def classify(sidecars: Sidecars, key: dict) -> str:
    """``line`` (exactly one), ``ambiguous``, ``dropped``, ``runtime-only``, ``undescribed``
    or ``unconverted``."""
    rows = sidecars.rows_holding(key)
    if rows is None:
        return "unconverted"
    if len(rows) > 1:
        return "ambiguous"
    if rows:
        return "line"
    if sidecars.is_dropped(key):
        return "dropped"
    return "runtime-only" if not sidecars.has_declared_slot(key) else "undescribed"


def _board_keys(record: dict) -> list[dict]:
    return [
        key
        for entity in entities_of(record)
        for line in entity.get("lines") or ()
        for key in line.get("keys") or ()
    ]


class AmbiguousKeys(TallyCheck):
    """Acting and board keys that do not resolve to exactly one line, by key kind."""

    NAMES = tuple(f"{p}-ambiguous[{k}]" for p in POSITIONS for k in KINDS) + tuple(
        f"{p}-undescribed-or-unconverted" for p in POSITIONS
    )

    def __init__(self, sidecars: Sidecars) -> None:
        super().__init__()
        self.sidecars = sidecars

    def feed(self, record: dict) -> None:
        self._tally("acting-keys", record.get("ability") or ())
        self._tally("board-lines", _board_keys(record))

    def _tally(self, position: str, keys) -> None:
        for key in keys:
            outcome = classify(self.sidecars, key)
            self.add(f"{position}-ambiguous[{key['trait_kind']}]", outcome == "ambiguous")
            unjoined = outcome in ("undescribed", "unconverted")
            self.add(f"{position}-undescribed-or-unconverted", unjoined)
