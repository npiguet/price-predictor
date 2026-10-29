"""The checks that read a record's events: repeats inside one record,
zone-change origins, clause attribution, causes and the event types seen."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
    Share,
    Thresholds,
    ceiling_finding,
    floor_finding,
    share_of,
)
from effects.domain.canonical_digest import canonical_digest
from effects.domain.event_schema import (
    CAUSE_BEARING_TYPES,
    AttributionKind,
    Event,
    EventType,
    attribution_kind,
)
from effects.domain.records import (
    EffectRecord,
    RecordKind,
)


class DuplicateEvents:
    """Byte-identical events inside one record: one outcome written twice,
    which the record-level check cannot see and which teaches the model that
    abilities deal their damage twice."""

    def __init__(self) -> None:
        #: Events repeating an earlier event of their own record.
        self.repeats = Share()
        self.by_type: Counter[EventType] = Counter()

    def add(self, record: EffectRecord) -> None:
        seen: set[bytes] = set()
        # Not all_events: a rewrite's unedited two sides would read as a
        # repeat on every healthy record.
        for event in record.payload.listed_events():
            digest = canonical_digest(event.as_dict())
            repeat = digest in seen
            self.repeats.note(repeat)
            if repeat:
                self.by_type[event.type] += 1
            seen.add(digest)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        repeats = self.repeats
        yield ceiling_finding(
            name="no record repeats an event inside itself",
            ceiling=limits.max_duplicate_event_rate,
            rate=repeats.rate,
            measured=lambda limit: (
                f"{repeats.hit}/{repeats.seen} events repeat another in the "
                f"same record ({repeats.rate:.1%}, {limit})"
            ),
            detail=(
                f"{event_type}: {count}"
                for event_type, count in self.by_type.most_common(EXAMPLES)
            ),
        )


class ZoneChanges:
    """Where a card came from, and how often "the stack" is where it went.

    Without ``from_zone`` a graveyard recursion reads like a token entering.
    Watched, because a card made rather than moved has no origin. A channel
    mostly ``to_zone=stack`` is reporting casts rather than the board.
    """

    def __init__(self) -> None:
        #: zone_change events carrying a ``from_zone``.
        self.with_from = Share()
        self.to_zone: Counter[str] = Counter()

    def add(self, record: EffectRecord) -> None:
        for event in record.payload.all_events():
            if event.type is EventType.ZONE_CHANGE:
                self.with_from.note(bool(event.params.get("from_zone")))
                self.to_zone[str(event.params.get("to_zone"))] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        with_from = self.with_from
        to_stack = self.to_zone.get("stack", 0)
        stack_rate = share_of(to_stack, with_from.seen)
        yield floor_finding(
            name="zone_change events say where the card came from",
            floor=limits.min_zone_change_from_zone_rate,
            seen=with_from.seen,
            rate=with_from.rate,
            measured=lambda limit: (
                f"{with_from.hit}/{with_from.seen} carry "
                f"from_zone ({with_from.rate:.1%}, {limit}); {to_stack} say "
                f"to_zone=stack ({stack_rate:.1%})"
            ),
            detail=(
                f"to_zone={zone}: {count}"
                for zone, count in self.to_zone.most_common(EXAMPLES)
            ),
        )


def resolution_events(record: EffectRecord) -> tuple[Event, ...]:
    """Resolution events, rated by how many say something: a sub-ability
    index, the root, or an explicit unresolved. ``absent`` says nothing, since
    a pre-sentinel writer used it for both root and unresolved."""
    if record.kind is not RecordKind.RESOLUTION:
        return ()
    return record.payload.listed_events()


def fork_events(record: EffectRecord) -> tuple[Event, ...]:
    """Every event on a fork record, rated apart because the fork collectors
    are different code, and a fork whose events name no clause cannot be
    compared clause by clause with the record it mirrors."""
    return record.payload.all_events() if record.fork else ()


class Attribution:
    """Whether one population of events (``events_of``) names the clause that
    produced it."""

    def __init__(
        self,
        name: str,
        noun: str,
        events_of: Callable[[EffectRecord], tuple[Event, ...]],
        floor_of: Callable[[Thresholds], float | None],
    ) -> None:
        self.name = name
        self.noun = noun
        self.events_of = events_of
        self.floor_of = floor_of
        self.events = 0
        self.attribution: Counter[AttributionKind] = Counter()

    def add(self, record: EffectRecord) -> None:
        for event in self.events_of(record):
            self.events += 1
            self.attribution[attribution_kind(event.attributed_to)] += 1

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        named = self.events - self.attribution.get(AttributionKind.ABSENT, 0)
        rate = share_of(named, self.events)
        yield floor_finding(
            name=self.name,
            floor=self.floor_of(limits),
            seen=self.events,
            rate=rate,
            measured=lambda limit: (
                f"{named}/{self.events} {self.noun} name a clause "
                f"({rate:.1%}, {limit})"
            ),
            detail=(
                f"{state}: {count}"
                for state, count in sorted(self.attribution.items())
            ),
        )


class CauseBearing:
    """Whether the events that declare a ``cause`` populate it.

    Over :data:`~effects.domain.event_schema.CAUSE_BEARING_TYPES` alone, since
    most types correctly never carry one. An empty ``cause`` also misleads the
    duplicate-event check: two attackers dealing the same damage to one player
    differ only by it.
    """

    def __init__(self) -> None:
        #: Per cause-bearing event type: how often it names a cause, of how often seen.
        self.by_type: dict[str, Share] = defaultdict(Share)

    def add(self, record: EffectRecord) -> None:
        for event in record.payload.all_events():
            if event.type in CAUSE_BEARING_TYPES:
                self.by_type[event.type.value].note(bool(event.params.get("cause")))

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        caused = Share.total(self.by_type.values())
        yield floor_finding(
            name="events that declare a cause name one",
            floor=limits.min_cause_rate,
            seen=caused.seen,
            rate=caused.rate,
            measured=lambda limit: (
                f"{caused.hit}/{caused.seen} events of the "
                f"{len(CAUSE_BEARING_TYPES)} cause-bearing types carry a cause "
                f"({caused.rate:.1%}, {limit})"
            ),
            detail=(
                f"{event_type}: {counts.hit}/{counts.seen}"
                for event_type, counts in sorted(self.by_type.items())
            ),
        )


class EventTypes:
    """Which of the declared event types the window contains.

    Watched: a short window legitimately misses rare types. It measures fires,
    where ``test_event_schema_completeness.py`` only finds types the Java
    source never references, so a type referenced from code that never runs
    shows up here alone.
    """

    def __init__(self) -> None:
        #: Every distinct event type the window named, anywhere in a record.
        self.seen: set[str] = set()

    def add(self, record: EffectRecord) -> None:
        self.seen.update(event.type.value for event in record.payload.all_events())

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        declared = {member.value for member in EventType}
        unseen = tuple(sorted(declared - self.seen))
        yield Finding(
            name="declared event types are observed in the window",
            ok=True,
            watched=True,
            measured=(
                f"{len(self.seen)}/{len(declared)} declared event types "
                "appeared at least once"
            ),
            detail=unseen,
        )
