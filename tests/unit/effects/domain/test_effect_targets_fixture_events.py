"""Per event type over the gen-3 fixture: how many real events set a target.

Each real event of a curated record is derived alone (the record with that one
event as its whole outcome), so a type whose events set nothing is seen as one
rather than hidden by a neighbour in the same record. A type the head reads sets
a target on most of its events; a type no entity column reads is listed, with
its reason, and sets none.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import EventType
from tests.unit.effects.builders import curated_fixture_records

#: Event types the fixture holds whose events set no entity or player target.
SETS_NO_TARGET = frozenset({
    # the aux heads read these (effects_created, turn_structure_change, the
    # created-objects group), not an entity column
    EventType.CONTINUOUS_EFFECT_CREATED, EventType.DELAYED_TRIGGER_CREATED,
    EventType.PHASE_SKIPPED, EventType.TOKEN_CREATED,
    # the head has no output for the game's end (model-layout.md)
    EventType.PLAYER_LOST, EventType.PLAYER_WON,
    # no output column reads them
    EventType.ABILITY_COUNTERED, EventType.CHOICE_MADE, EventType.DICE_ROLLED,
})
#: A type the head reads sets a target on at least this share of its events.
MIN_SHARE = 0.5


@pytest.fixture(scope="module")
def counts() -> tuple[Counter, Counter]:
    """``(events, events setting a target)`` per event type."""
    seen: Counter = Counter()
    setting: Counter = Counter()
    for record in curated_fixture_records():
        payload = record.payload
        if not hasattr(payload, "events") or payload.listed_events() != payload.happened_events():
            continue
        for event in payload.events:
            seen[event.type] += 1
            alone = derive_targets(replace(record, payload=replace(payload, events=(event,))))
            setting[event.type] += any(
                entry.fields or entry.unknown for entry in alone.values())
    return seen, setting


def test_the_types_that_set_no_target_are_exactly_the_listed_ones(counts):
    seen, setting = counts
    silent = {event_type for event_type in seen if not setting[event_type]}
    assert silent == SETS_NO_TARGET & set(seen), silent ^ SETS_NO_TARGET


def test_a_type_the_head_reads_sets_a_target_on_most_of_its_events(counts):
    seen, setting = counts
    thin = {event_type.value: (setting[event_type], seen[event_type])
            for event_type in set(seen) - SETS_NO_TARGET
            if setting[event_type] / seen[event_type] < MIN_SHARE}
    assert not thin, thin


def test_the_fixture_holds_enough_events_to_judge(counts):
    seen, _ = counts
    assert sum(seen.values()) > 300 and len(seen) >= 25, len(seen)
