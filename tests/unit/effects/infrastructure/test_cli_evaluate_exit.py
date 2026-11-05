"""A failed gate is reported, never an exit code that stops the sweep."""

from __future__ import annotations

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
        effect_model_store, "load_effect_checkpoint",
        lambda path: SimpleNamespace(provenance=SplitProvenance()),
    )
    args = build_parser().parse_args(["evaluate-effect-model"])

    assert args.func(args) == 2
    assert "records no curated corpus" in caplog.text


def test_a_corpus_not_wholly_gen3_or_a_checkpoint_of_another_layout_exits_one(
    monkeypatch,
) -> None:
    """contracts/cli.md: evaluate-effect-model reads gen-3 records and a
    gen-3 checkpoint only (FR-076)."""
    from effects.domain.effect_head_input import CheckpointLayoutError
    from effects.domain.shard_generation import (
        MixedGenerationsError,
        UnacceptedGenerationError,
    )

    for error in (UnacceptedGenerationError("gen-2 shards"),
                  MixedGenerationsError("gen-2 and gen-3 shards"),
                  CheckpointLayoutError("gen-2 layout")):
        def refuse(config, error=error):
            raise error

        monkeypatch.setattr(evaluate_effect_model, "run", refuse)
        args = build_parser().parse_args(["evaluate-effect-model"])
        assert args.func(args) == 1, type(error).__name__


def test_the_selection_set_is_found_where_training_finds_it() -> None:
    """No new flag: the reports directory is training's default."""
    from pathlib import Path

    from effects.application.evaluation_context import EvaluateEffectModelConfig
    from effects.domain.default_paths import DEFAULT_REPORTS_DIR
    from effects.infrastructure.cli import config_from_args

    args = build_parser().parse_args(["evaluate-effect-model"])
    config = config_from_args(EvaluateEffectModelConfig, args)
    assert config.reports_dir == Path(DEFAULT_REPORTS_DIR)
    assert not hasattr(args, "reports_dir")
