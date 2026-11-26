"""validate-corpus's trigger, zone-change and attribution checks."""

from __future__ import annotations

from effects.application.corpus_validation.findings import Thresholds
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.records import RecordKind, ResolutionPayload
from tests.unit.effects.application.validate_corpus_ports import validate_corpus
from tests.unit.effects.application.validate_corpus_support import (
    _combat,
    _healthy,
    _named,
    _resolution,
    _snapshot,
    _trigger,
    _TriggerEvent,
)


class TestTriggerNegatives:
    def test_a_window_of_positives_only_fails(self):
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", state=_snapshot(turn=7 + index))
            for index in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "draw negatives")
        assert not finding.ok
        assert "fired:not-fired" in finding.measured

    @staticmethod
    def _ended(game_id: str):
        from tests.unit.effects.builders import game_end_marker

        return game_end_marker(_combat(f"{game_id}.999", game_id=game_id,
                                       state=_snapshot(turn=30)))

    def test_a_finished_game_s_negatives_only_fail(self):
        """V5 #19: the ceiling alone let 0% fired pass, the positives the
        reservoir writes at the game's end all lost. Judged over games that
        ended in the window: an unfinished game's positives are still held."""
        game = "run.0-L1.0"
        negatives = [_trigger(f"run.0-L1.{index}", event=_TriggerEvent(fired=False), game_id=game,
                              state=_snapshot(turn=7)) for index in range(900, 960)]
        records = [r for r in _healthy() if r.kind not in (RecordKind.TRIGGER,
                                                           RecordKind.TURN_ACTION)] + negatives
        unfinished = _named(validate_corpus(records), "trigger records hold positives")
        assert unfinished.ok and unfinished.watched
        assert "0 trigger records of games that ended" in unfinished.measured
        finished = _named(validate_corpus([*records, self._ended(game)]),
                          "trigger records hold positives")
        assert not finished.ok
        assert "0:60" in finished.measured and "floor 5.0%" in finished.measured

    def test_a_balanced_window_passes_and_reports_the_ratio(self):
        finding = _named(validate_corpus(_healthy()), "draw negatives")
        assert finding.ok, finding.measured
        assert "2:2 fired:not-fired" in finding.measured

    def test_the_ratio_is_reported_per_evaluated_mode(self):
        """One mode drawing no negatives hides behind every other mode."""
        records = _healthy() + [
            _trigger(
                f"run.0-L1.{index}", event=_TriggerEvent(mode="Attacks"),
                state=_snapshot(turn=7 + index),
            )
            for index in range(900, 903)
        ]
        finding = _named(validate_corpus(records), "draw negatives")
        assert any(
            line.startswith("Attacks: 3:0") for line in finding.detail
        ), finding.detail

    def test_a_window_with_no_triggers_is_not_a_failure(self):
        records = [
            record for record in _healthy()
            if record.kind is not RecordKind.TRIGGER
        ]
        assert _named(validate_corpus(records), "draw negatives").ok


class TestZoneChanges:
    def _arrival(self, record_id: str, params: dict):
        return _resolution(
            record_id,
            payload=ResolutionPayload(events=(
                Event(type=EventType.ZONE_CHANGE, subjects=("E1",), params=params),
            )),
        )

    def test_the_from_zone_share_is_watched_by_default(self):
        records = _healthy() + [
            self._arrival(f"run.0-L1.{i}", {"to_zone": "graveyard"})
            for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "where the card came from")
        assert finding.watched
        assert finding.ok, "a watched number never fails a run"
        assert "carry from_zone" in finding.measured

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [
            self._arrival(f"run.0-L1.{i}", {"to_zone": "graveyard"})
            for i in range(900, 910)
        ]
        strict = Thresholds(min_zone_change_from_zone_rate=0.9)
        finding = _named(
            validate_corpus(records, strict), "where the card came from",
        )
        assert not finding.watched
        assert not finding.ok
        assert "floor 90.0%" in finding.measured

    def test_events_that_do_say_where_from_clear_the_floor(self):
        records = _healthy() + [
            self._arrival(
                f"run.0-L1.{i}",
                {"from_zone": "battlefield", "to_zone": "graveyard"},
            )
            for i in range(900, 910)
        ]
        strict = Thresholds(min_zone_change_from_zone_rate=0.9)
        assert _named(
            validate_corpus(records, strict), "where the card came from",
        ).ok

    def test_the_share_landing_on_the_stack_rides_along(self):
        records = _healthy() + [
            self._arrival(
                f"run.0-L1.{i}", {"from_zone": "hand", "to_zone": "stack"},
            )
            for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "where the card came from")
        assert "10 say to_zone=stack" in finding.measured


class TestAttribution:
    def _with(self, record_id, attributed_to):
        return _resolution(
            record_id,
            payload=ResolutionPayload(events=(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("E1",),
                    attributed_to=attributed_to,
                ),
            )),
        )

    def test_the_three_states_are_counted_apart_from_absent(self):
        records = _healthy() + [
            self._with("run.0-L1.900", "2"),
            self._with("run.0-L1.901", AttributionKind.ROOT),
            self._with("run.0-L1.902", AttributionKind.UNRESOLVED),
            self._with("run.0-L1.903", None),
        ]
        detail = " ".join(
            _named(validate_corpus(records), "name what produced them").detail
        )
        assert "sub_ability: 1" in detail
        assert "unresolved: 1" in detail
        assert "absent: 1" in detail

    def test_a_null_pointer_is_not_counted_as_the_root(self):
        """The ambiguity the sentinels end, asserted rather than assumed."""
        records = [self._with(f"run.0-L1.{i}", None) for i in range(900, 910)]
        finding = _named(validate_corpus(records), "name what produced them")
        assert "0/10 resolution events name a clause" in finding.measured

    def test_the_rate_is_watched_until_a_floor_is_given(self):
        records = [self._with(f"run.0-L1.{i}", None) for i in range(900, 910)]
        assert _named(validate_corpus(records), "name what produced them").watched
        strict = Thresholds(min_attributed_rate=0.9)
        judged = _named(
            validate_corpus(records, strict), "name what produced them",
        )
        assert not judged.watched
        assert not judged.ok
