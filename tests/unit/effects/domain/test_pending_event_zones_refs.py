"""The pending event's zones, defender, blockers and cause reach the input.

A "dies" or "leaves the battlefield" trigger fires on a zone change from the
battlefield to the graveyard, so ``[GLOBAL]`` reads the event's ``from_zone``
and ``to_zone``, one-hot over :data:`ZONES`. The ``defender``, each of the
``blockers`` and the ``cause`` are refs: a card slot reads whether its card is
one (a relation from ``[ACT]`` would need a ``LinkFamily`` in the connector),
and a player slot whether its player is the defender or the cause. The
collector writes ``cause`` as the ref of the object that caused the event
(``E74``, ``P1``), not a vocabulary.

No fixture record carries a defender or blockers, so those two cases build
their event by hand (an ``attackers_declared`` and a ``became_blocked``); the
rest edit a real record.
"""

from __future__ import annotations

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.event_schema import Event, EventType
from effects.domain.input_layout import player_features
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_pending_event_input import _surface, _with_event, _with_pending

#: A trigger on a zone change from the stack to the battlefield, caused by E52.
_ZONE_TRIGGER = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.24"


@pytest.fixture(scope="module")
def record():
    return next(found for found in map(record_from_dict, gen3_fixture_records())
                if found.record_id == _ZONE_TRIGGER)


def _global(rec):
    return _surface(rec).slots[0].features


def _pending(rec, entity_id):
    entity = rec.state.entity(entity_id)
    return card_features(entity, rec)[card_segments(entity, rec)["pending_event"]]


def _player(rec, player_id):
    return player_features(next(p for p in rec.state.players if p.id == player_id), rec)


def _attackers(rec, defender):
    return _with_event(rec, Event(
        type=EventType.ATTACKERS_DECLARED, subjects=(),
        params={"mode": "AttackersDeclared", "defender": defender}))


def _blocked(rec, blockers):
    return _with_event(rec, Event(
        type=EventType.BECAME_BLOCKED, subjects=("E1",),
        params={"mode": "AttackerBlocked", "blockers": blockers}))


def test_the_from_zone_reaches_the_global_slot(record):
    assert record.state.pending_event.params["from_zone"] == "stack"
    assert _global(record) != _global(_with_pending(record, from_zone="battlefield"))


def test_the_to_zone_reaches_the_global_slot(record):
    assert record.state.pending_event.params["to_zone"] == "battlefield"
    assert _global(record) != _global(_with_pending(record, to_zone="graveyard"))


def test_dying_reads_apart_from_leaving_for_exile(record):
    dies = _with_pending(record, from_zone="battlefield", to_zone="graveyard")
    exiled = _with_pending(record, from_zone="battlefield", to_zone="exile")
    assert _global(dies) != _global(exiled)


def test_a_card_reads_whether_it_is_the_defender(record):
    other = _attackers(record, "E74")
    assert _pending(other, "E74")[3] == 1.0
    assert _pending(other, "E1")[3] == 0.0
    assert _surface(other).slots != _surface(_attackers(record, "E1")).slots


def test_a_player_reads_whether_it_is_the_defender(record):
    assert _player(_attackers(record, "P1"), "P1") != _player(_attackers(record, "P0"), "P1")


def test_a_card_reads_whether_it_is_among_the_blockers(record):
    other = _blocked(record, ["E74", "E15"])
    assert [_pending(other, e)[4] for e in ("E74", "E15", "E1")] == [1.0, 1.0, 0.0]
    assert _surface(other).slots != _surface(_blocked(record, ["E74"])).slots


def test_a_card_reads_whether_it_is_the_cause(record):
    assert record.state.pending_event.params["cause"] == "E52"
    assert _pending(record, "E52")[5] == 1.0
    other = _with_pending(record, cause="E74")
    assert (_pending(other, "E52")[5], _pending(other, "E74")[5]) == (0.0, 1.0)
    assert _surface(record).slots != _surface(other).slots


def test_a_player_reads_whether_it_is_the_cause(record):
    assert _player(record, "P1") != _player(_with_pending(record, cause="P1"), "P1")
