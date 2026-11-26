"""validate-corpus's fork and probe checks: probe visibility, lossy forks, a
fork's turn against its mirror, and the clause a fork event names.
"""

from __future__ import annotations

from effects.application.corpus_validation.findings import Thresholds
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.records import CombatPayload, ResolutionPayload
from tests.unit.effects.application.validate_corpus_ports import validate_corpus
from tests.unit.effects.application.validate_corpus_support import (
    _combat,
    _healthy,
    _named,
    _resolution,
    _snapshot,
)


class TestProbeVisibility:
    def test_a_window_with_no_probes_says_so_without_failing(self):
        finding = _named(validate_corpus(_healthy()), "probe forks")
        assert finding.watched and finding.ok
        assert "0 damage-step probe forks" in finding.measured
        assert any("--probe-keywords" in line for line in finding.detail)

    def test_a_probe_fork_is_counted_and_its_keyword_named(self):
        records = _healthy() + [
            _combat(
                "run.0-L1.900", fork=True,
                payload=CombatPayload(
                    attackers=("E0",), probed_keyword="trample",
                    probed_entity="E0",
                ),
            ),
        ]
        finding = _named(validate_corpus(records), "probe forks")
        assert "1 damage-step probe forks" in finding.measured
        assert "trample 1" in finding.measured

    def test_an_intervention_is_counted_apart_from_a_probe(self):
        """The flag signatures differ, and so must the two counts."""
        records = _healthy() + [
            _resolution("run.0-L1.900", interventional=True, fork=True),
        ]
        finding = _named(validate_corpus(records), "probe forks")
        assert "0 damage-step probe forks, 1 interventional forks" in (
            finding.measured
        )


class TestLossyForks:
    """The share of fork records whose copy left something out, by kind:
    watched, since a lossy fork is kept and trained on."""

    def test_a_window_with_no_forks_reports_none(self):
        finding = _named(validate_corpus(_healthy()), "copy left something out")
        assert finding.watched and finding.ok
        assert finding.measured == (
            "0/0 fork records; by kind: none; 0/0 forks (at least; by game, turn and kind of"
            " fork); by kind: none")

    def test_the_share_is_counted_per_kind_over_every_fork_record(self):
        records = _healthy() + [
            _resolution("run.0-L1.900", interventional=True, fork=True,
                        copy_dropped=(("command", 2), ("remembered", 1))),
            _resolution("run.0-L1.901", interventional=True, fork=True,
                        copy_dropped=(("remembered", 1),)),
            _resolution("run.0-L1.902", interventional=True, fork=True),
            _resolution("run.0-L1.903", interventional=True, fork=True),
        ]
        finding = _named(validate_corpus(records), "copy left something out")
        assert finding.watched and finding.ok
        assert finding.measured == (
            "2/4 fork records; by kind: command 1/4 (25.0%), remembered 2/4 (50.0%); "
            "1/1 forks (at least; by game, turn and kind of fork); "
            "by kind: command 1/1 (100.0%), remembered 1/1 (100.0%)")

    def test_a_fork_of_many_records_is_one_fork(self):
        """A protection fork's ten records carry its copy's copy_dropped each:
        ten lossy records, one lossy fork; another game's fork is another."""
        lossy = [_resolution(f"run.0-L1.{900 + i}", interventional=True, fork=True,
                             copy_dropped=(("command", 1),)) for i in range(10)]
        clean = [_resolution("run.0-L1.950", game_id="run.0-L1.1", interventional=True,
                             fork=True)]
        finding = _named(validate_corpus(_healthy() + lossy + clean), "copy left something out")
        assert finding.measured.startswith("10/11 fork records; by kind: command 10/11 (90.9%); "
                                           "1/2 forks"), finding.measured
        assert finding.measured.endswith("by kind: command 1/2 (50.0%)"), finding.measured


class TestAForksTurnAgreesWithItsMirror:
    """The residue of the game_id check, named where it can be acted on.

    A probe re-runs one damage step from the state the real combat was in, so
    the fork and the record it mirrors describe one moment. When the fork's
    snapshot lags, the game_id check sees a backward turn jump and blames a
    game merge that never happened — which is how 3.9% of the smoke corpus's
    probe forks were reported for a whole pass.
    """

    def _probe(self, record_id: str, mirror: str, turn: int):
        return _combat(
            record_id, fork=True, mirror_of=mirror, state=_snapshot(turn=turn),
            payload=CombatPayload(attackers=("E0",), probed_keyword="trample"),
        )

    def _real(self, record_id: str, turn: int):
        return _combat(record_id, state=_snapshot(turn=turn))

    def test_a_fork_at_its_mirrors_turn_holds(self):
        records = _healthy() + [
            self._real("run.0-L1.900", 6),
            self._probe("run.0-L1.901", "run.0-L1.900", 6),
        ]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert finding.ok, finding.detail
        assert "0/1 forks disagree" in finding.measured

    def test_a_stale_fork_snapshot_is_named_with_both_turns(self):
        records = _healthy() + [
            self._real("run.0-L1.900", 9),
            self._probe("run.0-L1.901", "run.0-L1.900", 8),
        ]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert not finding.ok, finding.measured
        assert "1/1 forks disagree" in finding.measured
        assert any(
            "run.0-L1.901 at turn 8 mirrors run.0-L1.900 at turn 9" in line
            for line in finding.detail
        )

    def test_a_fork_ahead_of_its_mirror_is_caught_too(self):
        """Only lag has been observed, but the invariant is agreement."""
        records = _healthy() + [
            self._real("run.0-L1.900", 6),
            self._probe("run.0-L1.901", "run.0-L1.900", 7),
        ]
        assert not _named(
            validate_corpus(records), "agrees with the record it mirrors"
        ).ok

    def test_a_mirror_outside_a_cut_window_is_counted_apart_and_never_passed(
        self,
    ):
        """``--limit`` cuts the stream mid-game; the missing half is the
        reader's doing, not the collector's, so with no comparison at all the
        figure is watched rather than passed."""
        records = _healthy() + [self._probe("run.0-L1.901", "run.0-L1.900", 8)]
        finding = _named(validate_corpus(records, window_cut=True),
                         "agrees with the record it mirrors")
        assert finding.ok and finding.watched
        assert "no comparison made: all 1 mirrors fell outside the window" in finding.measured

    def test_a_mirror_nowhere_in_a_directory_read_whole_fails(self):
        """Nothing can be outside the window then: a mirror no record carries
        was never written, and 0/0 disagreements is no pass."""
        records = _healthy() + [self._probe("run.0-L1.901", "run.0-L1.900", 8)]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert not finding.ok, finding.measured
        assert "0/0 forks disagree" in finding.measured
        assert "1 mirrors fell outside the window (100.0%" in finding.measured

    def test_a_fork_written_before_the_record_it_mirrors_still_joins(self):
        """A shard may carry either half first, so the comparison waits for the
        end of the pass rather than for the next record."""
        records = _healthy() + [
            self._probe("run.0-L1.901", "run.0-L1.900", 8),
            self._real("run.0-L1.900", 9),
        ]
        assert not _named(
            validate_corpus(records), "agrees with the record it mirrors"
        ).ok


class TestForkEventsNameAClause:
    """Attribution on the records the fork collectors write, measured apart.

    The aggregate hid this: the observed records reached 84.8% naming a clause
    while every one of the 1,698 fork events in the same window said nothing,
    and the two averaged into a number that read as a channel warming up.
    """

    def _fork(self, record_id: str, attributed_to):
        return _resolution(
            record_id, interventional=True, fork=True,
            payload=ResolutionPayload(
                events=(
                    Event(
                        type=EventType.DAMAGE_DEALT, subjects=("E1",),
                        attributed_to=attributed_to,
                    ),
                ),
            ),
        )

    def test_a_window_with_no_forks_says_so_without_failing(self):
        finding = _named(validate_corpus(_healthy()), "fork records")
        assert finding.watched and finding.ok
        assert "0/0" in finding.measured

    def test_silent_fork_events_are_watched_by_default(self):
        records = _healthy() + [self._fork("run.0-L1.900", None)]
        finding = _named(validate_corpus(records), "fork records")
        assert finding.watched and finding.ok
        assert "0/1 events on fork records name a clause" in finding.measured
        assert "absent: 1" in finding.detail

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [self._fork("run.0-L1.900", None)]
        finding = _named(
            validate_corpus(records, Thresholds(min_fork_attributed_rate=0.9)),
            "fork records",
        )
        assert not finding.ok and not finding.watched

    def test_a_fork_naming_the_root_line_clears_the_floor(self):
        records = _healthy() + [self._fork("run.0-L1.900", AttributionKind.ROOT)]
        assert _named(
            validate_corpus(records, Thresholds(min_fork_attributed_rate=0.9)),
            "fork records",
        ).ok

    def test_an_explicit_unresolved_counts_as_naming_something(self):
        """Saying the pointer did not land is a fact; saying nothing is not."""
        records = _healthy() + [self._fork("run.0-L1.900", AttributionKind.UNRESOLVED)]
        finding = _named(validate_corpus(records), "fork records")
        assert "1/1 events on fork records name a clause" in finding.measured
