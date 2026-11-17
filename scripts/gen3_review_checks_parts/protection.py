"""Protection forks: does an attachment that leaves get an event naming it?"""

from __future__ import annotations

import json

from .model import Check, Line, entities_of, events_of, state_of

LEAVE_EVENTS = ("unattached", "zone_change", "destroyed", "sacrificed")


def _is_granted_protection(line: dict) -> bool:
    if line.get("kind") != "keyword" or line.get("keys"):
        return False
    return bool(line.get("granted")) and str(line.get("fragment", "")).startswith("Protection")


def _granted_protection(record: dict) -> tuple[str, str] | None:
    """The creature carrying a granted, unkeyed Protection line, and that line's fragment."""
    found = [
        (entity["id"], line["fragment"])
        for entity in entities_of(record)
        if entity.get("zone") == "battlefield"
        for line in entity.get("lines") or ()
        if _is_granted_protection(line)
    ]
    return found[-1] if found else None


def _attachments_to(record: dict) -> dict[str, str | None]:
    """Attachment id to host id on a live record."""
    return {e["id"]: e["attached_to"] for e in entities_of(record) if e.get("attached_to")}


def _still_attached(entity: dict | None, creature: str) -> bool:
    if entity is None:
        return False
    return entity.get("zone") == "battlefield" and entity.get("attached_to") == creature


class _Fork:
    """One protection fork: the attachments its creature held and what its records show of them."""

    def __init__(self, creature: str, attachments: list[str]) -> None:
        self.creature = creature
        self.attachments = attachments
        self.gone: set[str] = set()
        self.named: set[str] = set()

    def see(self, record: dict) -> None:
        by_id = {e["id"]: e for e in entities_of(record)}
        self.gone.update(
            a for a in self.attachments if not _still_attached(by_id.get(a), self.creature)
        )
        leaves = [e for e in events_of(record) if e.get("type") in LEAVE_EVENTS]
        self._name(" ".join(json.dumps(e) for e in leaves))

    def _name(self, text: str) -> None:
        self.named.update(a for a in self.attachments if f'"{a}"' in text)


def _is_protection_fork(record: dict) -> bool:
    if not record.get("fork") or record.get("directed"):
        return False
    return bool(record.get("interventional"))


class ProtectionForks(Check):
    """Protection forks whose attachment leaves with no event naming it."""

    def __init__(self) -> None:
        self.live: dict[str, dict[str, str | None]] = {}
        self.forks: dict[tuple, _Fork] = {}

    def feed(self, record: dict) -> None:
        if not record.get("fork"):
            if entities_of(record):
                self.live[record["game_id"]] = _attachments_to(record)
            return
        grant = _granted_protection(record) if _is_protection_fork(record) else None
        if grant is not None:
            self._fork_of(record, grant).see(record)

    def _fork_of(self, record: dict, grant: tuple[str, str]) -> _Fork:
        game = record["game_id"]
        key = (game, (state_of(record).get("global") or {}).get("turn"), *grant)
        if key not in self.forks:
            held = [a for a, host in self.live.get(game, {}).items() if host == grant[0]]
            self.forks[key] = _Fork(grant[0], held)
        return self.forks[key]

    def lines(self) -> list[Line]:
        held = [(fork, att) for fork in self.forks.values() for att in fork.attachments]
        unnamed = sum(1 for fork, att in held if att in fork.gone and att not in fork.named)
        return [Line("protection-forks-attachment-leaves-unnamed", unnamed, len(held))]
