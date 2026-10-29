"""Gate margins against their limits, and the verdict built from them."""

from __future__ import annotations

import numpy as np

from effects.application.evaluate_effect_model import not_implemented
from effects.application.gate_one import GateOneMetrics, evaluate_gate_one
from effects.application.gate_two import evaluate_gate_two, evaluate_keyword
from effects.application.geometry_checks import (
    CacheGeometry,
    gate_three_verdict,
)
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
)
from effects.domain.evaluation_report import (
    Bound,
    CheckName,
    CheckStatus,
    EvaluationReport,
    Margin,
    gate_verdict,
)
from tests.unit.effects.builders import keyword_score


class TestMargin:
    """One gate quantity against its limit, and the verdict over several."""

    def test_a_floor_is_breached_below_its_limit_and_held_at_it(self):
        assert Margin("k", "gain", 0.04, 0.05, Bound.FLOOR).breached
        assert not Margin("k", "gain", 0.05, 0.05, Bound.FLOOR).breached

    def test_a_ceiling_is_breached_above_its_limit_and_held_at_it(self):
        assert Margin("k", "cosine", 0.51, 0.5, Bound.CEILING).breached
        assert not Margin("k", "cosine", 0.5, 0.5, Bound.CEILING).breached

    def test_a_margin_that_cannot_be_measured_fails(self):
        margin = Margin("k", "gain", float("nan"), 0.05, Bound.FLOOR)
        assert margin.breached
        assert margin.failure == "gain not measured"

    def test_the_failure_shows_both_sides_in_their_formats(self):
        margin = Margin(
            "k", "top component", 0.41, 0.3, Bound.CEILING,
            value_format=".1%", limit_format=".0%", verb="explains",
        )
        assert margin.failure == "top component explains 41.0% > 30%"

    def test_the_verdict_names_every_breach_and_reports_every_value(self):
        result = gate_verdict(
            CheckName.GATE_1,
            (
                Margin("a", "first", 0.0, 0.05, Bound.FLOOR, value_format="+.3f"),
                Margin("b", "second", 0.1, 0.05, Bound.FLOOR),
                Margin("c", "third", float("nan"), 0.05, Bound.FLOOR),
            ),
            fail_prefix="it failed", pass_detail="it held",
        )
        assert result.status is CheckStatus.FAIL
        assert result.detail == "it failed: first +0.000 < 0.05; third not measured"
        assert list(result.values) == ["a", "b", "c"]

    def test_the_verdict_passes_when_every_margin_holds(self):
        result = gate_verdict(
            CheckName.GATE_3, (Margin("a", "first", 0.1, 0.05, Bound.FLOOR),),
            fail_prefix="it failed", pass_detail="it held",
        )
        assert (result.status, result.detail) == (CheckStatus.PASS, "it held")


class TestReport:
    def test_a_report_ships_when_no_gate_blocks(self):
        report = EvaluationReport()
        report.add(evaluate_gate_one(
            GateOneMetrics(0.70, 0.60, 0.90), GateOneMetrics(0.60, 0.50, 1.00),
        ))
        report.add(not_implemented(CheckName.ROLE_POLARITY))
        assert report.ships

    def test_a_blocking_gate_stops_it(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        assert not report.ships
        assert [c.name for c in report.blocking_failures] == ["gate-3"]

    def test_gate_two_never_stops_it(self):
        report = EvaluationReport()
        report.keyword_verdicts = [
            evaluate_keyword(row, keyword_score(1, 0))
            for row in DAMAGE_STEP_KEYWORDS
        ]
        report.add(evaluate_gate_two(report.keyword_verdicts))
        assert report.ships

    def test_the_render_names_every_check_and_the_verdict(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        rendered = report.render()
        assert "gate-3" in rendered
        assert "GATES FAILED: gate-3" in rendered

    def test_a_passing_gate_still_prints_every_figure_it_measured(self):
        """The design record needs all three gate-1 margins, pass or fail."""
        report = EvaluationReport()
        report.add(evaluate_gate_one(
            GateOneMetrics(0.70, 0.60, 0.90), GateOneMetrics(0.60, 0.50, 1.00),
        ))
        rendered = report.render()
        assert "affected_gate_f1_gain=+0.1000" in rendered
        assert "zone_outcome_accuracy_gain=+0.1000" in rendered
        assert "poisson_deviance_reduction=+0.1000" in rendered

    def test_a_failing_gate_prints_the_figure_that_held_too(self):
        report = EvaluationReport()
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        rendered = report.render()
        assert "mean_pairwise_cosine=" in rendered
        assert "top_component_share=" in rendered
