"""validate-corpus's checks of what a resolution reports: outcomes, costs,
snapshot tiers, the collection mode and empty mana.
"""

from __future__ import annotations

from effects.application.corpus_validation.findings import Thresholds
from effects.application.validate_corpus import validate_corpus
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    CollectionMode,
    Costs,
    EffectRecord,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.state_snapshot import InclusionTier
from tests.unit.effects.application.validate_corpus_support import (
    _activation,
    _healthy,
    _named,
    _resolution,
    _snapshot,
)

_ALL_TIERS = frozenset(InclusionTier)


class TestOutcome:
    def test_one_outcome_everywhere_fails(self):
        """The activation half hardcoded `resolved`, so 100% of them said so."""
        records = [
            record for record in _healthy()
            if not (
                isinstance(record.payload, ResolutionPayload)
                and record.payload.outcome is ResolutionOutcome.FIZZLED
            )
        ]
        finding = _named(validate_corpus(records), "outcome")
        assert not finding.ok
        assert "1 distinct" in finding.measured
        assert "resolved" in finding.measured

    def test_two_outcomes_pass_and_the_counts_are_reported(self):
        finding = _named(validate_corpus(_healthy()), "outcome")
        assert finding.ok
        assert "fizzled 2" in finding.measured
        assert "resolved 2" in finding.measured

    def test_the_effect_halves_carry_it(self):
        """The outcome is on the effect half (FR-042b), the cost half carries
        none: one value over effect halves fails, two pass."""
        def halves(*outcomes):
            return [
                _resolution(f"run.0-L1.{900 + i}", payload=ResolutionPayload(
                    events=(), outcome=outcome, outcome_written=True))
                for i, outcome in enumerate(outcomes)
            ]

        literal = _named(validate_corpus(halves(ResolutionOutcome.RESOLVED,
                                                ResolutionOutcome.RESOLVED)), "outcome")
        assert not literal.ok and "1 distinct over 2" in literal.measured
        varied = _named(validate_corpus(halves(ResolutionOutcome.RESOLVED,
                                               ResolutionOutcome.FIZZLED)), "outcome")
        assert varied.ok and "fizzled 1" in varied.measured


class TestCosts:
    def test_costs_empty_on_every_record_fails(self):
        """Four cost lists were never once populated in 14.7M records."""
        records = [
            _activation(f"run.0-L1.{index}", costs=Costs())
            for index in range(500, 510)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert not finding.ok
        assert "0/10" in finding.measured

    def test_the_report_names_each_dead_channel(self):
        finding = _named(validate_corpus(_healthy()), "cost fields")
        assert finding.ok
        assert "sacrificed 0" in finding.measured
        assert "mana_by_color 4" in finding.measured

    def test_a_channel_every_deck_pays_reading_zero_fails(self):
        """`tapped` was populated zero times in 14.7M records.

        The whole cost object was not empty — mana came off the printed cost —
        so a check that only asks whether anything was ever paid passes the
        corpus that motivated it.
        """
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 500)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert not finding.ok
        assert "never populated over 300 activation records: tapped" in (
            " ".join(finding.detail)
        )

    def test_a_window_too_small_to_judge_says_so_rather_than_failing(self):
        """Scarcity and a dead channel look identical until the window is big."""
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 210)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert finding.ok
        assert "under the 200" in " ".join(finding.detail)

    def test_the_evidence_floor_is_configurable(self):
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 210)
        ]
        strict = Thresholds(min_cost_evidence=5)
        assert not _named(validate_corpus(records, strict), "cost fields").ok


class TestSnapshotTiers:
    def test_a_tier_depth_chosen_per_collector_fails(self):
        """Tier 4 appeared on the interventional records and nowhere else.

        That makes ``state.tiers`` a perfect predictor of a field the schema
        forbids the model to see, which is a leak rather than a depth choice.
        """
        records = _healthy() + [
            _resolution(
                "run.0-L1.600", interventional=True, fork=True,
                state=_snapshot(tiers=_ALL_TIERS),
            ),
        ]
        finding = _named(validate_corpus(records), "tier depth")
        assert not finding.ok
        assert "2 distinct tier vectors" in finding.measured
        assert any("[1, 2, 3, 4]" in line for line in finding.detail)

    def test_one_vector_everywhere_passes(self):
        assert _named(validate_corpus(_healthy()), "tier depth").ok


class TestCollectionMode:
    def test_a_degraded_record_fails_the_run(self):
        """One stock checkout in the pool silently loses four record kinds."""
        records = _healthy() + [
            _resolution("run.0-L1.900", mode=CollectionMode.DEGRADED),
        ]
        finding = _named(validate_corpus(records), "patched mode")
        assert not finding.ok
        assert "degraded 1" in finding.measured
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_patched_window_passes_and_reports_the_count(self):
        finding = _named(validate_corpus(_healthy()), "patched mode")
        assert finding.ok
        assert "0/18 not patched" in finding.measured


class TestEmptyMana:
    """Story 2, acceptance 12: the count of mana_produced events naming no mana."""

    @staticmethod
    def _mana(record_id: str, mana) -> EffectRecord:
        params = {} if mana is None else {"mana_by_color": mana}
        return _resolution(record_id, payload=ResolutionPayload(events=(
            Event(type=EventType.MANA_PRODUCED, subjects=("P0",), params=params),)))

    def test_the_empty_ones_are_counted_and_watched(self):
        records = [self._mana("r1", {"G": 1}), self._mana("r2", {}),
                   self._mana("r3", None), self._mana("r4", {"W": 0, "U": 0})]
        finding = _named(validate_corpus(records), "name the mana they made")
        assert finding.watched
        assert finding.measured.startswith("3/4 mana_produced events have an empty")
        assert finding.detail == ("resolution: 3",)
