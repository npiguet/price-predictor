"""The three record-quality rules build-corpus applies (FR-148)."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.record_quality import (
    EVENT_FLOOD,
    NO_ABILITY,
    UNATTRIBUTED,
    quality_defect,
)
from effects.domain.records import CombatPayload, RecordKind, ResolutionPayload


def _event(attributed_to=None):
    return Event(type=EventType.DAMAGE_DEALT, subjects=("P0",), attributed_to=attributed_to)


def test_a_clean_resolution_record_has_no_defect(make_record):
    assert quality_defect(make_record()) is None


def test_a_resolution_record_with_no_ability_is_a_defect(make_record):
    assert quality_defect(make_record(ability=())) == NO_ABILITY
    assert quality_defect(make_record(ability=None)) == NO_ABILITY


def test_a_combat_record_needs_no_ability(make_record):
    record = make_record(kind=RecordKind.COMBAT, payload=CombatPayload(events=(_event(),)))
    assert quality_defect(record) is None


def test_an_unresolved_attribution_is_a_defect(make_record):
    payload = ResolutionPayload(events=(_event("root"), _event("unresolved")))
    assert quality_defect(make_record(payload=payload)) == UNATTRIBUTED


def test_an_unknown_attribution_is_not_a_defect(make_record):
    payload = ResolutionPayload(events=(_event(None),))
    assert quality_defect(make_record(payload=payload)) is None


def test_too_many_events_is_a_defect(make_record):
    payload = ResolutionPayload(events=tuple(_event("root") for _ in range(65)))
    assert quality_defect(make_record(payload=payload)) == EVENT_FLOOD
    assert quality_defect(make_record(payload=payload), max_events=100) is None


def test_no_ability_wins_over_the_other_reasons(make_record):
    payload = ResolutionPayload(events=tuple(_event("unresolved") for _ in range(65)))
    assert quality_defect(make_record(ability=(), payload=payload)) == NO_ABILITY
