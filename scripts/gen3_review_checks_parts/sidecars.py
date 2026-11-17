"""Sidecars read once each, and the audit of a converted tree for keys on two lines."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .model import Line, key_tuple

TREES = ("cardsfolder", "tokenscripts")


class Sidecars:
    """The converted tree's sidecars and their text, read on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, dict | None] = {}

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

    def rows_holding(self, key: dict) -> list[dict] | None:
        """Every sidecar line declaring ``key``; ``None`` when its script has no sidecar."""
        data = self.get(key["script_file"])
        if data is None:
            return None
        wanted = key_tuple(key)
        return [
            line
            for line in data["lines"]
            if any(key_tuple(p) == wanted for p in line["provenance"])
        ]

    def is_dropped(self, key: dict) -> bool:
        data = self.get(key["script_file"])
        if data is None:
            return False
        return any(key_tuple(k) == key_tuple(key) for k in data["dropped_keys"])

    def has_declared_slot(self, key: dict) -> bool:
        """Whether ``key``'s (face, trait kind) slot reaches its index in the sidecar."""
        data = self.get(key["script_file"])
        slot = (key["face"], key["trait_kind"])
        indices = [
            k["index_within_kind"]
            for k in _declared_keys(data)
            if (k["face"], k["trait_kind"]) == slot
        ]
        return bool(indices) and key["index_within_kind"] <= max(indices)

    def line_text(self, script_file: str, line: dict) -> str:
        data = self.get(script_file)
        index = line["line_index"]
        return data["_text"][index] if data and 0 <= index < len(data["_text"]) else ""


def _declared_keys(data: dict | None) -> list[dict]:
    if data is None:
        return []
    return [k for line in data["lines"] for k in line["provenance"]] + list(data["dropped_keys"])


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
