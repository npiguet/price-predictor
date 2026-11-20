"""validate-corpus's event checks: duplicate records and events, events that
declare a cause, and which event types a corpus holds.
"""

from __future__ import annotations

from effects.application.corpus_validation.findings import Finding, Thresholds
from effects.application.validate_corpus import validate_corpus
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.records import EffectRecord, ResolutionPayload, RewritePayload
from tests.unit.effects.application.validate_corpus_support import (
    _healthy,
    _named,
    _resolution,
    _rewrite,
    _snapshot,
    _trigger,
)


def _record_with_event(event_type: EventType, record_id: str = "r0", **overrides):
    """A minimal record whose only event is one of ``event_type``.

    For the event-type coverage check, which cares only which types appear
    anywhere in a record, not what kind of record carries them. No ``params``
    are set, so this is valid for any ``event_type`` regardless of what its
    schema row declares.
    """
    fields: dict = {
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=event_type, subjects=("E1",),
                    attributed_to=AttributionKind.ROOT,
                ),
            ),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _event_types(records: list[EffectRecord]) -> Finding:
    return _named(validate_corpus(records), "event types")


class TestDuplicates:
    def test_a_flood_of_identical_records_in_one_game_fails(self):
        """N triggers of one mode against one event rendered byte-identically."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", state=_snapshot(turn=4))
            for index in range(700, 720)
        ]
        finding = _named(validate_corpus(records), "duplicates")
        assert not finding.ok
        assert "trigger" in " ".join(finding.detail)

    def test_identical_records_in_different_games_are_not_duplicates(self):
        """Two games can legitimately reach the same board and the same answer."""
        records = _healthy() + [
            _trigger("run.0-L1.800", game_id="run.0-L1.7", state=_snapshot(turn=4)),
        ]
        assert _named(validate_corpus(records), "duplicates").ok

    def test_the_identity_fields_do_not_hide_a_duplicate(self):
        """Two records differing only in id and timestamp are one observation."""
        records = _healthy() + [
            _trigger(
                f"run.0-L1.{index}", state=_snapshot(turn=4),
                timestamp=f"2026-09-09T11:00:{index - 700:02d}.000000Z",
            )
            for index in range(700, 720)
        ]
        assert not _named(validate_corpus(records), "duplicates").ok


class TestDuplicateEventsInsideOneRecord:
    """13.68% of the smoke corpus's events, invisible to the record check."""

    def _twice(self, index: int):
        """One record whose single outcome is written twice.

        Each record sits on its own turn so the *records* stay distinct: the
        point of this class is that the record-level duplicate check sees
        nothing, and it would see these if they were copies of each other.
        """
        event = Event(
            type=EventType.DAMAGE_DEALT, subjects=("E1",),
            params={"amount": 3}, attributed_to=AttributionKind.ROOT,
        )
        return _resolution(
            f"run.0-L1.{index}", state=_snapshot(turn=index - 890),
            payload=ResolutionPayload(events=(event, event)),
        )

    def test_one_outcome_written_twice_fails(self):
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "repeats an event")
        assert not finding.ok
        assert "damage_dealt" in " ".join(finding.detail)

    def test_the_record_level_check_cannot_see_it(self):
        """Why this check exists at all: the records themselves are unique."""
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        assert _named(validate_corpus(records), "exact duplicates").ok

    def test_two_events_differing_only_in_subject_are_not_duplicates(self):
        """Two damage events on two creatures are two outcomes."""
        records = _healthy() + [
            _resolution(
                "run.0-L1.900",
                payload=ResolutionPayload(events=(
                    Event(type=EventType.DAMAGE_DEALT, subjects=("E1",)),
                    Event(type=EventType.DAMAGE_DEALT, subjects=("E2",)),
                )),
            ),
        ]
        assert _named(validate_corpus(records), "repeats an event").ok

    def test_a_rewrite_carrying_one_event_twice_is_not_a_duplicate(self):
        """A pre-contract shard wrote the same event into both slots.

        Those records are still in the corpus, so the event-duplication check
        still has to ignore a rewrite's own pair — the *rewrite* checks are
        what report the copy, and they report it as the collector defect it is
        rather than as a duplicated outcome.
        """
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 3})
        records = _healthy() + [
            _rewrite(
                "run.0-L1.900",
                payload=RewritePayload(incoming=same, outgoing=same),
            ),
        ]
        assert _named(validate_corpus(records), "repeats an event").ok

    def test_the_rate_is_configurable(self):
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        loose = Thresholds(max_duplicate_event_rate=0.9)
        assert _named(validate_corpus(records, loose), "repeats an event").ok


class TestEventsThatDeclareACauseNameOne:
    """The identity channel the duplicate-event check leans on.

    Two attackers dealing 1 to the same player differ in nothing but their
    cause, so an empty one turns two real outcomes into one outcome written
    twice — the collector is right not to dedupe them and the duplicate check
    is right to complain, and only this number says which side to fix.
    """

    def _zone_change(self, record_id: str, cause):
        params = {"from_zone": "battlefield", "to_zone": "graveyard"}
        if cause is not None:
            params["cause"] = cause
        return _resolution(
            record_id,
            payload=ResolutionPayload(
                events=(
                    Event(
                        type=EventType.ZONE_CHANGE, subjects=("E1",),
                        params=params, attributed_to=AttributionKind.ROOT,
                    ),
                ),
            ),
        )

    def test_a_missing_cause_is_watched_by_default(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", None)]
        finding = _named(validate_corpus(records), "declare a cause")
        assert finding.watched and finding.ok
        # Four of the healthy window's trigger records carry a zone_change of
        # their own, and none of them names a cause either.
        assert "0/5 events" in finding.measured

    def test_a_populated_cause_is_counted(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", "E7")]
        finding = _named(validate_corpus(records), "declare a cause")
        assert "1/5 events" in finding.measured
        assert "zone_change: 1/5" in finding.detail

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", None)]
        finding = _named(
            validate_corpus(records, Thresholds(min_cause_rate=0.5)),
            "declare a cause",
        )
        assert not finding.ok and not finding.watched

    def test_a_type_whose_row_declares_no_cause_is_not_measured(self):
        """Every type *may* carry a cause and most correctly never do, so a
        rate over the whole vocabulary would read near zero and mean nothing.
        The healthy window's resolutions are damage_dealt, which declares
        ``source`` instead, so only its four zone_changes are counted."""
        finding = _named(validate_corpus(_healthy()), "declare a cause")
        assert "0/4 events" in finding.measured
        assert not any("damage_dealt" in line for line in finding.detail)


class TestEventTypeCoverage:
    """Which of the declared event types a corpus actually contains.

    Watched, not judged: a short window legitimately misses rare types, and a
    floor nobody has calibrated would fail every smoke run. What it must not
    do is stay silent — 29 of the vocabulary's types were unreachable for the
    life of the project and no report ever said so.

    This measures *fires*, which the static guard in
    ``test_event_schema_completeness.py`` cannot: that guard scans the
    connector's Java source for a reference to each type's constant, and
    ``damage_prevented``/``spell_copied`` were referenced from dead code
    while firing zero times across a 10.1M-record corpus. A type this check
    never saw is one of those two gaps or a type too rare for the window —
    this check cannot tell them apart, and does not pretend to.
    """

    def test_it_reports_the_types_a_corpus_never_showed(self):
        records = [
            _record_with_event(EventType.ZONE_CHANGE, "r0"),
            _record_with_event(EventType.DAMAGE_DEALT, "r1"),
        ]

        finding = _event_types(records)

        assert finding.watched is True
        assert "2/" in finding.measured
        assert "energy_change" in finding.detail

    def test_a_type_seen_twice_is_not_listed_as_unseen(self):
        records = [
            _record_with_event(EventType.ZONE_CHANGE, "r0"),
            _record_with_event(EventType.ZONE_CHANGE, "r1"),
        ]

        finding = _event_types(records)

        assert "1/" in finding.measured
        assert "zone_change" not in finding.detail

    def test_it_never_fails_the_run(self):
        """Watched means never a verdict, however sparse the window is."""
        finding = _event_types([_record_with_event(EventType.ZONE_CHANGE)])
        assert finding.ok is True

    def test_the_full_vocabulary_reports_nothing_unseen(self):
        records = [
            _record_with_event(event_type, f"r{index}")
            for index, event_type in enumerate(EventType)
        ]

        finding = _event_types(records)

        assert finding.detail == ()
        total = len(list(EventType))
        assert f"{total}/{total}" in finding.measured
