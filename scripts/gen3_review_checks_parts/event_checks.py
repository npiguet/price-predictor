"""Checks over single events: trigger subjects, skipped phases, mana attribution, directed forks."""

from __future__ import annotations

from collections import defaultdict

from .model import (
    Check,
    Line,
    TallyCheck,
    events_of,
    is_effect_half,
    payload_of,
    state_of,
)

SUBJECTLESS_MODES = ("DamageDealtOnce", "DamageAll")
MANA_EVENTS = ("tapped", "mana_produced")


class TriggerSubjects(TallyCheck):
    """`DamageDealtOnce` and `DamageAll` trigger records with no subject."""

    NAMES = tuple(f"trigger-records-without-subject[{mode}]" for mode in SUBJECTLESS_MODES)

    def feed(self, record: dict) -> None:
        if record.get("kind") != "trigger":
            return
        event = payload_of(record).get("event") or {}
        mode = (event.get("params") or {}).get("mode")
        if mode in SUBJECTLESS_MODES:
            self.add(f"trigger-records-without-subject[{mode}]", not event.get("subjects"))


class PhaseSkipped(TallyCheck):
    """`phase_skipped` events whose phase is the placeholder `?`."""

    NAMES = ("phase-skipped-with-unknown-phase",)

    def feed(self, record: dict) -> None:
        for event in events_of(record):
            if event.get("type") == "phase_skipped":
                unknown = (event.get("params") or {}).get("phase") == "?"
                self.add("phase-skipped-with-unknown-phase", unknown)


class ManaAttribution(TallyCheck):
    """Events of a mana record (an effect half with no outcome) with a null `attributed_to`."""

    NAMES = tuple(f"mana-record-events-without-attribution[{kind}]" for kind in MANA_EVENTS)

    def feed(self, record: dict) -> None:
        events = events_of(record)
        is_mana = is_effect_half(record) and payload_of(record).get("outcome") is None
        if not is_mana or not any(e.get("type") == "mana_produced" for e in events):
            return
        for event in events:
            if event.get("type") in MANA_EVENTS:
                self.add(
                    f"mana-record-events-without-attribution[{event['type']}]",
                    event.get("attributed_to") is None,
                )


class DirectedForks(Check):
    """Directed-fork records per rule: the distinct forks (game, turn, variant, operation)."""

    def __init__(self) -> None:
        self.records: dict[str, int] = defaultdict(int)
        self.forks: dict[str, set] = defaultdict(set)

    def feed(self, record: dict) -> None:
        tag = record.get("directed")
        if not tag:
            return
        turn = (state_of(record).get("global") or {}).get("turn")
        for rule in record.get("aimed_at") or ():
            self.records[rule] += 1
            fork = (record.get("game_id"), turn, tag.get("variant"), tag.get("operation"))
            self.forks[rule].add(fork)

    def lines(self) -> list[Line]:
        return [
            Line(f"directed-forks[{rule}]", len(self.forks[rule]), count)
            for rule, count in sorted(self.records.items())
        ]
