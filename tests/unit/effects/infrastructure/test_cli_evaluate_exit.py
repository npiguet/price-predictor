"""A failed gate is reported, never an exit code that stops the sweep."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from effects.application import evaluate_effect_model
from effects.application.geometry_checks import CacheGeometry, gate_three_verdict
from effects.domain.evaluation_report import EvaluationReport
from effects.infrastructure.cli import build_parser


def test_a_failed_gate_still_exits_zero(monkeypatch, capsys) -> None:
    report = EvaluationReport()
    report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
    monkeypatch.setattr(evaluate_effect_model, "run", lambda config: report)
    args = build_parser().parse_args(["evaluate-effect-model"])

    assert args.func(args) == 0
    assert "GATES FAILED: gate-3" in capsys.readouterr().out


def test_a_checkpoint_that_recorded_no_corpus_exits_two(monkeypatch, caplog) -> None:
    """Every gate reads the curated strata; nothing else names its training games."""
    from effects.domain.split_provenance import SplitProvenance
    from effects.infrastructure import effect_model_store

    monkeypatch.setattr(
        effect_model_store, "load_for_inference",
        lambda path, **_: (
            SimpleNamespace(provenance=SplitProvenance()),
            Path("vocab.txt"), Path("keywords.json"),
        ),
    )
    args = build_parser().parse_args(["evaluate-effect-model"])

    assert args.func(args) == 2
    assert "records no curated corpus" in caplog.text
