"""The evaluator's report of the checks it never built (FR-116)."""

from __future__ import annotations

from effects.application.evaluate_effect_model import not_implemented
from effects.domain.evaluation_report import CheckName, CheckStatus, EvaluationReport


class TestUnimplementedChecks:
    """FR-116 names three checks that were never built.

    The corpus holds forks and mana records, so a line blaming missing records
    would send an operator looking for data that is already there.
    """

    def test_each_reports_not_implemented_and_blocks_nothing(self):
        for name in (
            CheckName.MATCHED_REAL_VS_FORK, CheckName.PROBE_DIFF,
            CheckName.ROLE_POLARITY,
        ):
            result = not_implemented(name)
            assert result.status is CheckStatus.NOT_IMPLEMENTED
            assert not result.blocks
            assert result.detail.startswith("not implemented")

    def test_the_report_marks_them_apart_from_a_skip(self):
        report = EvaluationReport()
        report.add(not_implemented(CheckName.PROBE_DIFF))
        assert report.render().startswith("[todo] probe-diff: not implemented")
