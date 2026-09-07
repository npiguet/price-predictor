"""Per-kind and per-stratum scoring.

The two rules that make the reported numbers mean something: metrics condition
on affected entities (most entities on most boards are untouched, so an
unconditioned accuracy is dominated by easy negatives), and the zone accuracy is
class-balanced (``stayed`` dominates, so an unbalanced one reads as high for a
model that answers "stayed" every time).
"""

from __future__ import annotations

import math

import pytest

from effects.application.evaluate_effect_model import Stratum
from effects.application.evaluation_scoring import (
    Predictions,
    StratifiedReport,
    count_valued_fields,
    observed_gate,
    stratify,
)
from effects.domain.effect_model import FIELDS_BY_NAME, ZONE_OUTCOMES, FieldType
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)


def _record(*events) -> EffectRecord:
    return EffectRecord(
        record_id="r.0.1", run_id="r", timestamp="t", game_id="r.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION,
        actor_player="P0", state=_SNAPSHOT,
        payload=ResolutionPayload(events=tuple(events)),
    )


class TestPredictions:
    def test_gate_f1_is_one_for_a_perfect_gate(self):
        p = Predictions(gate_predicted=[1, 0, 1], gate_observed=[1, 0, 1])
        assert p.gate_f1() == pytest.approx(1.0)

    def test_gate_f1_is_nan_for_an_empty_population(self):
        assert math.isnan(Predictions().gate_f1())

    def test_zone_accuracy_is_class_balanced(self):
        """A model that answers "stayed" every time must not score high."""
        stayed = ZONE_OUTCOMES.index("stayed")
        died = ZONE_OUTCOMES.index("died")
        always_stayed = Predictions(
            zone_predicted=[stayed] * 10,
            zone_observed=[stayed] * 9 + [died],
        )
        # Unbalanced accuracy would be 0.9; balanced it is 0.5.
        assert always_stayed.zone_accuracy() == pytest.approx(0.5)

    def test_zone_accuracy_is_one_when_every_class_is_right(self):
        p = Predictions(zone_predicted=[0, 1, 2], zone_observed=[0, 1, 2])
        assert p.zone_accuracy() == pytest.approx(1.0)

    def test_a_class_with_no_observations_is_skipped(self):
        p = Predictions(zone_predicted=[0, 0], zone_observed=[0, 0])
        assert p.zone_accuracy() == pytest.approx(1.0)

    def test_deviance_is_zero_for_an_exact_prediction(self):
        p = Predictions(count_predicted=[3.0, 5.0], count_observed=[3.0, 5.0])
        assert p.mean_deviance() == pytest.approx(0.0, abs=1e-9)

    def test_the_three_metrics_pack_into_gate_one_metrics(self):
        p = Predictions(
            gate_predicted=[1, 0], gate_observed=[1, 0],
            zone_predicted=[0], zone_observed=[0],
            count_predicted=[2.0], count_observed=[2.0],
        )
        metrics = p.as_gate_one_metrics()
        assert metrics.affected_gate_f1 == pytest.approx(1.0)
        assert metrics.zone_outcome_accuracy == pytest.approx(1.0)


class TestConditioning:
    def test_a_conditional_field_is_recorded_only_where_the_gate_fired(self):
        """Predicting a zone outcome for an untouched permanent was not asked."""
        report = StratifiedReport()
        report.observe(
            "resolution-effect", Stratum.UNIQUE_TEXT,
            gate_predicted=0.9, gate_observed=0.0,
            zone_predicted=1, zone_observed=1,
        )
        assert len(report.overall.zone_observed) == 0
        assert len(report.overall.gate_observed) == 1

    def test_an_affected_entity_contributes_to_both(self):
        report = StratifiedReport()
        report.observe(
            "resolution-effect", Stratum.UNIQUE_TEXT,
            gate_predicted=0.9, gate_observed=1.0,
            zone_predicted=1, zone_observed=1,
            count_predicted=3.0, count_observed=3.0,
        )
        assert len(report.overall.zone_observed) == 1
        assert len(report.overall.count_observed) == 1

    def test_the_gate_records_every_entity_affected_or_not(self):
        report = StratifiedReport()
        for observed in (0.0, 1.0, 0.0):
            report.observe(
                "combat", Stratum.UNIQUE_TEXT,
                gate_predicted=0.5, gate_observed=observed,
            )
        assert len(report.overall.gate_observed) == 3


class TestStratification:
    def _stratify(self, record, **overrides):
        defaults = {
            "training_texts": {"seen text"},
            "training_api_types": {"Pump", "Destroy"},
            "numeric_range": (0.0, 5.0),
            "text_of": lambda r: "unseen text",
            "api_types_of": lambda r: [],
            "numbers_of": lambda r: [],
        }
        defaults.update(overrides)
        return stratify(record, **defaults)

    def test_a_text_the_model_trained_on_is_shared_text(self):
        assert self._stratify(
            _record(), text_of=lambda r: "seen text",
        ) is Stratum.SHARED_TEXT

    def test_a_text_it_never_saw_is_unique_text(self):
        assert self._stratify(_record()) is Stratum.UNIQUE_TEXT

    def test_a_number_outside_the_training_range_extrapolates(self):
        assert self._stratify(
            _record(), numbers_of=lambda r: [9.0],
        ) is Stratum.NUMERIC_EXTRAPOLATION

    def test_a_number_inside_the_range_does_not(self):
        assert self._stratify(
            _record(), numbers_of=lambda r: [3.0],
        ) is Stratum.UNIQUE_TEXT

    def test_familiar_apis_in_an_unfamiliar_combination(self):
        assert self._stratify(
            _record(), api_types_of=lambda r: ["Pump", "Destroy"],
        ) is Stratum.NOVEL_COMBINATION

    def test_an_unfamiliar_api_is_unique_text_not_a_novel_combination(self):
        assert self._stratify(
            _record(), api_types_of=lambda r: ["Pump", "Bushido"],
        ) is Stratum.UNIQUE_TEXT

    def test_shared_text_wins_over_the_other_three(self):
        """A memorized text is a memorized text whatever else is true of it."""
        assert self._stratify(
            _record(), text_of=lambda r: "seen text",
            numbers_of=lambda r: [99.0],
        ) is Stratum.SHARED_TEXT


class TestCountValuedFields:
    def test_it_covers_every_count_and_signed_delta(self):
        names = set(count_valued_fields())
        assert "damage_taken" in names
        assert "life_delta" in names
        assert "cards_drawn" in names

    def test_it_excludes_categoricals_and_bits(self):
        names = set(count_valued_fields())
        assert "zone_outcome" not in names
        assert "tap_state" not in names
        assert "keywords_gained" not in names

    def test_every_name_is_a_real_field_of_the_right_type(self):
        for name in count_valued_fields():
            assert FIELDS_BY_NAME[name].type in (
                FieldType.COUNT, FieldType.SIGNED_DELTA,
            )


class TestObservedGate:
    def test_an_entity_an_event_names_is_affected(self):
        record = _record(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
        )
        assert observed_gate(record, "E1") == 1.0

    def test_an_entity_no_event_names_is_not(self):
        record = _record(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
        )
        assert observed_gate(record, "E2") == 0.0


class TestRendering:
    def _report(self) -> StratifiedReport:
        report = StratifiedReport()
        for observed in (1.0, 0.0):
            report.observe(
                "combat", Stratum.UNIQUE_TEXT,
                gate_predicted=observed, gate_observed=observed,
                zone_predicted=1, zone_observed=1,
                count_predicted=2.0, count_observed=2.0,
            )
        return report

    def test_the_table_names_every_kind_it_scored(self):
        assert "combat" in self._report().render("full")

    def test_the_state_only_floor_is_reported_beside_each_kind(self):
        """A number is only interpretable against what no ability info gets."""
        rendered = self._report().render("full", floor=self._report())
        assert "state-only floor" in rendered

    def test_strata_with_observations_are_listed(self):
        assert "[unique-text]" in self._report().render("full")

    def test_an_empty_report_renders_without_raising(self):
        assert StratifiedReport().render("full") == "full:"
