"""Gate margins against their limits, and the verdict built from them."""

from __future__ import annotations

from effects.domain.evaluation_report import Bound, CheckName, CheckStatus, Margin, gate_verdict


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
