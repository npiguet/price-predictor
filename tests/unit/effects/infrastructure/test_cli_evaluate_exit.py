"""A failed gate is reported, never an exit code that stops the sweep."""

from __future__ import annotations

import numpy as np

from effects.application import evaluate_effect_model
from effects.application.evaluate_effect_model import EvaluationReport, evaluate_gate_three
from effects.infrastructure.cli import build_parser


def test_a_failed_gate_still_exits_zero(monkeypatch, capsys) -> None:
    report = EvaluationReport()
    report.add(evaluate_gate_three(np.ones((50, 8))))
    monkeypatch.setattr(evaluate_effect_model, "run", lambda config: report)
    args = build_parser().parse_args(["evaluate-effect-model"])

    assert args.func(args) == 0
    assert "GATES FAILED: gate-3" in capsys.readouterr().out
