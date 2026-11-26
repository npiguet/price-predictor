"""A failed gate is reported, never an exit code that stops the sweep."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from effects.application import evaluate_effect_model
from effects.application.geometry_checks import CacheGeometry, gate_three_verdict
from effects.domain.evaluation_report import EvaluationReport
from tests.unit.effects.cli_args import parse_cli


def test_a_failed_gate_still_exits_zero(monkeypatch, capsys) -> None:
    def run(config, report: EvaluationReport) -> EvaluationReport:
        report.add(gate_three_verdict(CacheGeometry.of(np.ones((50, 8)))))
        return report

    monkeypatch.setattr(evaluate_effect_model, "run", run)
    args = parse_cli(["evaluate-effect-model"])

    assert args.func(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "GATES FAILED: gate-3 (not measured: gate-1)"
    assert sum("] gate-3:" in line for line in lines) == 1


def test_a_checkpoint_that_recorded_no_corpus_exits_one(monkeypatch, caplog) -> None:
    """Every gate reads the curated strata; nothing else names its training games."""
    from effects.domain.split_provenance import SplitProvenance
    from effects.infrastructure import effect_model_store

    monkeypatch.setattr(
        effect_model_store, "load_effect_checkpoint",
        lambda path: SimpleNamespace(provenance=SplitProvenance()),
    )
    args = parse_cli(["evaluate-effect-model"])

    assert args.func(args) == 1
    assert "records no curated corpus" in caplog.text


def test_a_corpus_not_wholly_gen3_or_a_checkpoint_of_another_layout_exits_one(
    monkeypatch,
) -> None:
    """contracts/cli.md: evaluate-effect-model reads gen-3 records and a
    gen-3 checkpoint only (FR-076)."""
    from effects.domain.checkpoint_layout import CheckpointLayoutError
    from effects.domain.shard_generation import (
        OlderGenerationError,
        UnknownGenerationError,
    )

    for error in (OlderGenerationError("the record is gen-2"),
                  UnknownGenerationError("generation 4"),
                  CheckpointLayoutError("gen-2 layout")):
        def refuse(config, report, error=error):
            raise error

        monkeypatch.setattr(evaluate_effect_model, "run", refuse)
        args = parse_cli(["evaluate-effect-model"])
        assert args.func(args) == 1, type(error).__name__


def test_the_selection_set_is_found_where_training_finds_it() -> None:
    """``--reports-dir`` is the directory training reads it from: both name it."""
    from pathlib import Path

    from effects.application.evaluation_context import EvaluateEffectModelConfig
    from effects.infrastructure.cli import config_from_args

    args = parse_cli(["evaluate-effect-model", "--reports-dir", "reports/frozen"])
    config = config_from_args(EvaluateEffectModelConfig, args)
    assert config.reports_dir == Path("reports/frozen")
    train = parse_cli(["train-effect-model", "--corpus", "c", "--reports-dir", "reports/frozen"])
    assert Path(args.reports_dir) == Path(train.reports_dir)


def _stub_everything_but_the_geometry(monkeypatch) -> None:
    """``run`` over fakes: the gates answer, the load steps and the other
    reported checks return nothing, so the geometry checks are the one thing
    a test makes raise."""
    from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

    def gate(name, status, **values):
        return CheckResult(name, status, "detail", values)

    context = SimpleNamespace(
        config=SimpleNamespace(abilities_root="abilities", output_dir="out",
                               win_rates="rates"),
        main=SimpleNamespace(provenance=SimpleNamespace(withheld_keyword="flying",
                                                        held_out_cards=())),
    )
    stubs = {
        "load_context": lambda config: context,
        "load_baselines": lambda config, main: {},
        "_sidecar_index": lambda context: None,
        "_load_cache": lambda context, sidecars, variant=None: None,
        "run_gate_three": lambda cached, root: gate(CheckName.GATE_3, CheckStatus.PASS),
        "_gate_one": lambda context, variants: gate(CheckName.GATE_1, CheckStatus.PASS),
        "run_gate_two": lambda context: SimpleNamespace(
            verdicts=[], check=gate(CheckName.GATE_2, CheckStatus.REPORTED)),
        "run_breakdowns": lambda context: [],
        "run_reporting_half": lambda context: [],
        "check_ward": lambda cached: gate(CheckName.WARD_CANARY, CheckStatus.REPORTED),
        "check_nearest_neighbours": lambda cached, queries: gate(
            CheckName.NEAREST_NEIGHBOUR, CheckStatus.REPORTED),
        "check_decodability": lambda cached, rates, held_out_cards: gate(
            CheckName.DECODABILITY, CheckStatus.REPORTED),
        "check_baseline_geometry": lambda cached, other, variant: gate(
            CheckName.baseline_comparison(variant), CheckStatus.REPORTED),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(evaluate_effect_model, name, stub)


def test_a_umap_check_that_raises_leaves_the_gate_lines_printed_and_exits_zero(
    monkeypatch, capsys,
) -> None:
    _stub_everything_but_the_geometry(monkeypatch)

    def draw(cached, output_dir):
        raise RuntimeError("numba could not compile")

    monkeypatch.setattr(evaluate_effect_model, "check_umap", draw)
    args = parse_cli(["evaluate-effect-model"])

    assert args.func(args) == 0
    lines = capsys.readouterr().out.splitlines()
    for gate in ("gate-1", "gate-2", "gate-3"):
        assert any(line.startswith("[") and f"] {gate}:" in line for line in lines)
    assert "[skip] umap: RuntimeError: numba could not compile" in lines
    assert lines[-1] == "SHIPS"
    assert sum("] gate-3:" in line for line in lines) == 1


def test_both_blocking_gates_skipped_prints_gates_not_measured_and_exits_zero(
    monkeypatch, capsys,
) -> None:
    from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

    _stub_everything_but_the_geometry(monkeypatch)
    for name, gate in (("run_gate_three", CheckName.GATE_3), ("_gate_one", CheckName.GATE_1)):
        monkeypatch.setattr(
            evaluate_effect_model, name,
            lambda *a, gate=gate: CheckResult(gate, CheckStatus.SKIPPED, "no cache"),
        )
    monkeypatch.setattr(evaluate_effect_model, "check_umap", lambda cached, out: CheckResult(
        CheckName.UMAP, CheckStatus.REPORTED, "drawn"))
    args = parse_cli(["evaluate-effect-model"])

    assert args.func(args) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "GATES NOT MEASURED: gate-1, gate-3"


def test_a_variant_checkpoint_of_another_variant_exits_one(monkeypatch, caplog) -> None:
    """``identity=<a full model>`` would score gate 1 against the full model."""
    from effects.application.evaluation_context import BaselineVariantError

    def refuse(config, report):
        raise BaselineVariantError("identity.pt is a 'full' checkpoint")

    monkeypatch.setattr(evaluate_effect_model, "run", refuse)
    args = parse_cli(["evaluate-effect-model"])
    assert args.func(args) == 1
    assert "is a 'full' checkpoint" in caplog.text
