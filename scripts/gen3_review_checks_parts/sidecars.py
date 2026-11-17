"""Sidecars read once each, and the audit of a converted tree for keys on two lines."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from effects.domain.provenance import ProvenanceKey
from effects.infrastructure.sidecar_io import SidecarCache

from .model import Line, key_tuple

TREES = ("cardsfolder", "tokenscripts")
UNDESCRIBED = "undescribed"


class Sidecars:
    """The converted tree's sidecars and their text, read on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, dict | None] = {}
        self.cache = SidecarCache({tree: root / tree for tree in TREES})

    def get(self, script_file: str) -> dict | None:
        if script_file not in self._cache:
            self._cache[script_file] = self._read(script_file)
        return self._cache[script_file]

    def _read(self, script_file: str) -> dict | None:
        tree, _, rel = script_file.partition("/")
        path = self.root / tree / (rel[: -len(".txt")] + ".provenance.json")
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        text = (self.root / tree / rel).read_text(encoding="utf-8")
        data["_text"] = text.split("\n")
        return data

    def resolution_of(self, key: dict) -> str:
        """The join's answer for a record's key (``KeyResolution`` value), or
        ``undescribed`` where the sidecar does not describe it."""
        try:
            return self.cache.join(ProvenanceKey.from_dict(key)).resolution.value
        except KeyError:
            return UNDESCRIBED

    def board_slots(self, key: dict) -> list[dict]:
        """The keys a board line of ``key`` reads as slots (``board_keys``)."""
        provenance = ProvenanceKey.from_dict(key)
        try:
            slots = self.cache.get(key["script_file"]).board_keys(provenance)
        except KeyError:
            return [key]
        return [slot.as_dict() for slot in slots]

    def line_text(self, script_file: str, line: dict) -> str:
        data = self.get(script_file)
        index = line["line_index"]
        return data["_text"][index] if data and 0 <= index < len(data["_text"]) else ""


def _has_doubly_declared_key(data: dict) -> bool:
    counts = Counter(key_tuple(k) for line in data["lines"] for k in line["provenance"])
    return any(n > 1 for n in counts.values())


def keys_declared_by_two_lines(root: Path) -> Line:
    """Cards of the tree with a key declared by two lines of one face (one tree audit)."""
    flagged = total = 0
    for tree in TREES:
        for path in sorted((root / tree).rglob("*.provenance.json")):
            total += 1
            flagged += _has_doubly_declared_key(json.loads(path.read_text(encoding="utf-8")))
    return Line("keys-declared-by-two-lines", flagged, total)
