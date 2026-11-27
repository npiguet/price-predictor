"""What the pending event is reaches the input.

A ``trigger`` record's fired bit and a ``rewrite`` record's outgoing event
depend on the pending event: "whenever CARDNAME deals combat damage" fires
on the event's ``source`` and ``combat``, a ``state_flag_change`` is a phase,
a room entered or an attacker unblocked only by its ``mode``, and a
replacement's outgoing amount (3 becomes 4) is the incoming one rewritten.
The ``[GLOBAL]`` slot reads the event's mode, one-hot over
:data:`PENDING_EVENT_MODES`, and whether it is combat damage; a card slot
reads whether its card is the event's subject and whether its source, a
player slot whether its player is the subject. A rewrite record's incoming
event is its pending event: the collector writes the two equal.

Each case builds two surfaces of one fixture record that differ only in the
one fact.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.input_layout import player_features
from effects.domain.input_vocabularies import PENDING_EVENT_MODES
from effects.domain.records import RecordKind
from effects.domain.surface_builder import SurfaceBuilder
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: A beginning-of-phase trigger: ``state_flag_change`` of mode ``Phase`` on P1.
_PHASE_TRIGGER = f"{_RUN}.0-mwqvqdyrb9c0.0"
#: A combat damage rewrite: E1 deals 1 combat damage to E44.
_COMBAT_DAMAGE = f"{_RUN}.2-mwqvqdxrusnv.631"
#: A combat damage rewrite: E9 deals 2 combat damage to E76.
_TWO_DAMAGE = f"{_RUN}.3-mwqvu8zeq2sj.3226"
#: A life-gain rewrite: P0 gains 1 life.
_LIFE_GAIN = f"{_RUN}.2-mwqvwtly1zvq.2412"


@pytest.fixture(scope="module")
def fixture_records():
    return list(map(record_from_dict, gen3_fixture_records()))


@pytest.fixture(scope="module")
def records(fixture_records):
    return {record.record_id: record for record in fixture_records}


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def _with_pending(record, **params):
    """The record with its pending event's params updated, and a rewrite's
    incoming event with it, which the collector writes equal."""
    pending = record.state.pending_event
    event = replace(pending, params={**pending.params, **params})
    return _with_event(record, event)


def _with_event(record, event):
    state = replace(record.state, pending_event=event)
    payload = record.payload
    if record.kind is RecordKind.REWRITE:
        payload = replace(payload, incoming=event)
    return replace(record, state=state, payload=payload)


def _pending_part(record, entity_id: str) -> tuple[float, ...]:
    """A card's pending-event part: pending, subject, source (three of six)."""
    entity = record.state.entity(entity_id)
    return card_features(entity, record)[card_segments(entity, record)["pending_event"]][:3]


def _player(record, player_id: str) -> tuple[float, ...]:
    return player_features(next(p for p in record.state.players if p.id == player_id), record)


def _global(record) -> tuple[float, ...]:
    return _surface(record).slots[0].features


def test_the_pending_events_mode_reaches_the_global_slot(records):
    record = records[_PHASE_TRIGGER]
    assert record.state.pending_event.params["mode"] == "Phase"
    begin = _with_pending(record, mode="BeginPhase")
    assert _global(record) != _global(begin)


def test_every_listed_mode_reads_its_own_column(records):
    record = records[_PHASE_TRIGGER]
    readings = {_global(_with_pending(record, mode=mode)) for mode in PENDING_EVENT_MODES}
    assert len(readings) == len(PENDING_EVENT_MODES)
    assert _global(_with_pending(record, mode="NoSuchMode")) not in readings


def test_combat_damage_reads_apart_from_other_damage(records):
    record = records[_COMBAT_DAMAGE]
    assert record.state.pending_event.params["combat"] is True
    assert _global(record) != _global(_with_pending(record, combat=False))


def test_a_card_reads_whether_it_is_the_pending_events_source(records):
    record = records[_COMBAT_DAMAGE]
    assert _pending_part(record, "E1") == (1.0, 0.0, 1.0)
    assert _pending_part(record, "E44") == (1.0, 1.0, 0.0)
    other = _with_pending(record, source="E44")
    assert _pending_part(other, "E1") == (1.0, 0.0, 0.0)
    assert _pending_part(other, "E44") == (1.0, 1.0, 1.0)
    assert _surface(record).slots != _surface(other).slots


def test_a_rewrites_incoming_event_is_its_pending_event(fixture_records):
    rewrites = [record for record in fixture_records if record.kind is RecordKind.REWRITE]
    assert rewrites
    assert all(record.state.pending_event == record.payload.incoming for record in rewrites)


def test_a_rewrites_incoming_amount_reaches_the_input(records):
    """The outgoing amount is the incoming one rewritten."""
    record = records[_TWO_DAMAGE]
    assert record.payload.incoming.params["amount"] == 2
    assert _global(record) != _global(_with_pending(record, amount=3))


def test_a_rewrites_incoming_subject_reaches_the_input(records):
    """A player subject too, which no card slot's subject bit covers."""
    record = records[_LIFE_GAIN]
    incoming = record.payload.incoming
    assert incoming.subjects == ("P0",)
    other = _with_event(record, replace(incoming, subjects=("P1",)))
    assert _player(record, "P0") != _player(other, "P0")
    assert _player(record, "P1") != _player(other, "P1")
    assert _surface(record).slots != _surface(other).slots
