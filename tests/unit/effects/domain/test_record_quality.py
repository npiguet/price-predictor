"""The two record-quality rules build-corpus applies (FR-148)."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.record_quality import (
    EVENT_FLOOD,
    NO_ABILITY,
    has_unattributed_events,
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


def test_an_unresolved_attribution_is_not_a_defect(make_record):
    """The bracket collector records the events that happened inside the
    ability's own resolution. Attribution to a clause can fail while the
    outcome is still that ability's: on the real corpus 12.3% of resolution
    records with an acting ability have every event stamped ``unresolved`` and
    none of them is a dump, and another 1% carry a single unattributed side
    effect — a state-based-action death, a ``choice_made``, a ``tapped``.
    Refusing them threw away the "died" zone outcome along with the rest.
    """
    payload = ResolutionPayload(events=(_event("root"), _event("unresolved")))
    assert quality_defect(make_record(payload=payload)) is None


def test_an_unknown_attribution_is_not_a_defect(make_record):
    payload = ResolutionPayload(events=(_event(None),))
    assert quality_defect(make_record(payload=payload)) is None


def test_too_many_events_is_a_defect(make_record):
    payload = ResolutionPayload(events=tuple(_event("root") for _ in range(65)))
    assert quality_defect(make_record(payload=payload)) == EVENT_FLOOD
    assert quality_defect(make_record(payload=payload), max_events=100) is None


def test_a_max_events_of_zero_means_no_cap_rather_than_refuse_everything(make_record):
    """Zero admits everything, the way ``CapHeap`` reads ``--text-cap 0``.

    ``build-corpus`` lists ``--max-events-per-record 0`` among the flags zero
    is a real setting for. Read as a literal ceiling it refused every record
    carrying a single event, so a build launched with it wrote an almost empty
    corpus and still exited 0.
    """
    payload = ResolutionPayload(events=tuple(_event("root") for _ in range(65)))
    assert quality_defect(make_record(payload=payload), max_events=0) is None


def test_no_ability_wins_over_the_event_flood(make_record):
    payload = ResolutionPayload(events=tuple(_event("root") for _ in range(65)))
    assert quality_defect(make_record(ability=(), payload=payload)) == NO_ABILITY


def test_a_combat_record_with_an_unresolved_event_survives(make_record):
    """Nothing resolves in a damage step, so the collector stamps every combat
    event ``unresolved`` by design — the cause lives in ``cause``. Refusing on
    that took 98.9% of the combat records in the real corpus."""
    payload = CombatPayload(events=(_event("unresolved"),))
    assert quality_defect(make_record(kind=RecordKind.COMBAT, payload=payload)) is None


def test_has_unattributed_events_sees_an_unresolved_event_of_any_kind(make_record):
    """The watch statistic the refusal became. Any kind, so the manifest's
    count is over the whole corpus rather than over the kinds a rule happened
    to reach."""
    assert has_unattributed_events(
        make_record(payload=ResolutionPayload(events=(_event("root"), _event("unresolved"))))
    )
    assert has_unattributed_events(
        make_record(kind=RecordKind.COMBAT, payload=CombatPayload(events=(_event("unresolved"),)))
    )


def test_has_unattributed_events_is_false_for_a_fully_attributed_record(make_record):
    assert not has_unattributed_events(
        make_record(payload=ResolutionPayload(events=(_event("root"), _event(None))))
    )
    assert not has_unattributed_events(make_record(payload=ResolutionPayload()))
