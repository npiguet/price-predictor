"""What every check shares: the report line, the tally base class and record predicates."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple


class Line(NamedTuple):
    """One report line: a check's name, the records it flagged and the records it looked at."""

    name: str
    count: int
    total: int


class Check:
    """A check is fed every record in shard order and reports its lines at the end."""

    def feed(self, record: dict) -> None:
        raise NotImplementedError

    def lines(self) -> list[Line]:
        raise NotImplementedError


class TallyCheck(Check):
    """A check that keeps a flagged/looked-at pair per name; ``NAMES`` always report."""

    NAMES: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.pairs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for name in self.NAMES:
            self.pairs[name] = [0, 0]

    def add(self, name: str, flagged: bool) -> None:
        pair = self.pairs[name]
        pair[0] += int(flagged)
        pair[1] += 1

    def lines(self) -> list[Line]:
        return [Line(name, pair[0], pair[1]) for name, pair in sorted(self.pairs.items())]


def is_fork(record: dict) -> bool:
    return bool(record.get("fork")) or bool(record.get("interventional"))


def is_effect_half(record: dict) -> bool:
    return record.get("kind") == "resolution" and record.get("moment") == "resolution"


def is_forced_half(record: dict) -> bool:
    """An interventional fork's effect half with no partner cost half: a forced play."""
    if record.get("directed") or not is_effect_half(record):
        return False
    if record.get("link_id") is not None:
        return False
    return bool(record.get("fork") and record.get("interventional"))


def payload_of(record: dict) -> dict:
    return record.get("payload") or {}


def events_of(record: dict) -> list[dict]:
    return payload_of(record).get("events") or []


def state_of(record: dict) -> dict:
    return record.get("state") or {}


def entities_of(record: dict) -> list[dict]:
    return state_of(record).get("entities") or []


def phase_of(record: dict) -> str | None:
    return (state_of(record).get("global") or {}).get("phase")


def first_key(record: dict) -> dict:
    """The record's first acting key, or an empty dict."""
    keys = record.get("ability") or []
    return keys[0] if keys else {}


def key_tuple(key: dict) -> tuple:
    """A key's identity within its script file; ``built`` is part of it."""
    built = key.get("built")
    return (key["face"], key["trait_kind"], key["index_within_kind"], key.get("option"),
            None if built is None else (built["kind"], built["index"]))
