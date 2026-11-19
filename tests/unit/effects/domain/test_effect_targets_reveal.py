"""``card_revealed``'s targets: the event names the player who revealed, and
only a library's reveal sets ``library_events``' ``reveal`` bit.

Each test edits the gen-3 fixture's Elfhame Sanctuary resolution, whose one
``card_revealed`` (from the hand, naming nobody) is the shape the collector
wrote before the event named its revealer.
"""

from __future__ import annotations

import functools
from dataclasses import replace

from effects.domain.effect_head_layout import LIBRARY_EVENTS
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import EventType
from effects.domain.records import EffectRecord

_ELFHAME = "9442f17f-cb56-4893-8d4a-66b5538de180.0-mwhr12znmj0t.1"
#: Elfhame Sanctuary's controller, who revealed the basic land it searched for.
_REVEALER = "P1"
_REVEAL = LIBRARY_EVENTS.index("reveal")


@functools.cache
def _elfhame() -> EffectRecord:
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return next(record for record in read_shard(GEN3_RECORDS) if record.record_id == _ELFHAME)


def _revealed(from_zone: str) -> EffectRecord:
    """The fixture record with its reveal naming the revealer, from ``from_zone``."""
    record = _elfhame()
    events = tuple(
        replace(event, subjects=(_REVEALER,), params={**event.params, "from_zone": from_zone})
        if event.type is EventType.CARD_REVEALED else event
        for event in record.payload.events
    )
    return replace(record, payload=replace(record.payload, events=events))


def _reveal_bit(record: EffectRecord, player: str) -> float | None:
    library_events = derive_targets(record)[player].fields.get("library_events")
    return None if library_events is None else library_events[_REVEAL]


def test_the_fixture_reveal_names_nobody_from_the_hand():
    reveals = [e for e in _elfhame().payload.events if e.type is EventType.CARD_REVEALED]
    assert [(e.subjects, e.params["from_zone"]) for e in reveals] == [((), "hand")]


def test_a_library_reveal_sets_the_reveal_bit_on_the_revealer():
    assert _reveal_bit(_revealed("library"), _REVEALER) == 1.0


def test_a_hand_reveal_sets_no_reveal_bit_on_the_revealer():
    targets = derive_targets(_revealed("hand"))
    assert targets[_REVEALER].affected
    library_events = targets[_REVEALER].fields.get("library_events")
    assert library_events is None or library_events[_REVEAL] == 0.0


def test_a_hand_reveal_is_counted_as_read_by_no_column():
    assert derive_targets(_revealed("hand")).unsupervised == {"card_revealed_off_library": 1}
    assert not derive_targets(_revealed("library")).unsupervised


def test_a_reveal_from_no_named_zone_sets_no_reveal_bit():
    record = _revealed("hand")
    events = tuple(
        replace(event, params={k: v for k, v in event.params.items() if k != "from_zone"})
        if event.type is EventType.CARD_REVEALED else event
        for event in record.payload.events
    )
    stripped = replace(record, payload=replace(record.payload, events=events))
    assert _reveal_bit(stripped, _REVEALER) in (None, 0.0)
