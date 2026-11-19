"""Where the evaluator's paths come from.

``--win-rates`` reaches the decodability battery (T108, Story 6 scenario 7),
and the variant tree defaults to the one the checkpoint's corpus recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.evaluation_context import EvaluateEffectModelConfig
from effects.domain.evaluation_report import EvaluationReport
from effects.infrastructure.cli import build_parser
from tests.unit.effects.cli_args import cli_args
from tests.unit.effects.configs import evaluation_config
from tests.unit.effects.domain.test_corpus_manifest import manifest


def _config_from(argv, monkeypatch) -> EvaluateEffectModelConfig:
    seen = {}

    def fake_run(config):
        seen["config"] = config
        return EvaluationReport()

    monkeypatch.setattr("effects.application.evaluate_effect_model.run", fake_run)
    args = build_parser().parse_args(cli_args(["evaluate-effect-model", *argv]))
    assert args.func(args) == 0
    return seen["config"]


def test_the_flag_names_the_table_the_battery_reads(monkeypatch, tmp_path):
    table = tmp_path / "rates.txt"
    config = _config_from(["--win-rates", str(table)], monkeypatch)
    assert config.win_rates == table


@pytest.mark.parametrize("flag", [
    "--checkpoint", "--abilities-root", "--win-rates", "--reports-dir",
])
def test_the_paths_a_run_reads_have_no_default(flag, capsys):
    """Artifact-layout rule 2: none names a pre-layout path, and the sealed
    pipeline's table is not read unless the command names it."""
    argv = cli_args(["evaluate-effect-model"])
    del argv[argv.index(flag):argv.index(flag) + 2]
    with pytest.raises(SystemExit) as refused:
        build_parser().parse_args(argv)
    assert refused.value.code == 2
    assert flag in capsys.readouterr().err


# ── defaults the corpus manifest supplies ────────────────────────────────


def test_the_variant_tree_defaults_to_the_one_the_corpus_recorded():
    """Without it every variant card's ability key resolves to no text."""
    from effects.application.evaluation_context import with_corpus_defaults

    recorded = manifest(variant_scripts="output/effects/variant-scripts/")
    config = with_corpus_defaults(evaluation_config(), recorded)
    assert config.variant_scripts == Path("output/effects/variant-scripts/")


def test_an_explicit_variant_tree_or_none_recorded_is_kept():
    from effects.application.evaluation_context import with_corpus_defaults

    chosen = evaluation_config(variant_scripts=Path("elsewhere"))
    recorded = manifest(variant_scripts="output/effects/variant-scripts/")
    assert with_corpus_defaults(chosen, recorded).variant_scripts == Path("elsewhere")
    blank = manifest(variant_scripts="")
    assert with_corpus_defaults(evaluation_config(), blank).variant_scripts is None


# ── where the report goes and where the selection set is read ────────────


def test_the_pictures_go_beside_the_checkpoint_by_default(monkeypatch, tmp_path):
    checkpoint = tmp_path / "run" / "latest.pt"
    config = _config_from(["--checkpoint", str(checkpoint)], monkeypatch)
    assert config.output is None
    assert config.output_dir == tmp_path / "run" / "latest.evaluation"


def test_output_names_the_report_directory(monkeypatch, tmp_path):
    report = tmp_path / "reports" / "2026-11-20-rule-mask-hard-evaluation"
    config = _config_from(["--output", str(report)], monkeypatch)
    assert config.output_dir == report


def test_reports_dir_names_where_the_selection_set_is_read(monkeypatch, tmp_path):
    frozen = tmp_path / "reports" / "2026-11-18-selection-set"
    assert _config_from(["--reports-dir", str(frozen)], monkeypatch).reports_dir == frozen


def test_the_run_directory_follows_output():
    from effects.infrastructure import cli

    spec = cli.RUN_MANIFESTS["evaluate-effect-model"]
    default = build_parser().parse_args(
        cli_args(["evaluate-effect-model", "--checkpoint", "models/effects/gen3/m/run/latest.pt"]))
    assert spec.run_dir(default) == Path("models/effects/gen3/m/run/latest.evaluation")
    chosen = build_parser().parse_args(cli_args([
        "evaluate-effect-model", "--output", "output/effects/gen3/reports/run",
        "--reports-dir", "output/effects/gen3/reports/frozen",
    ]))
    assert spec.run_dir(chosen) == Path("output/effects/gen3/reports/run")
    assert Path("output/effects/gen3/reports/frozen") in spec.input_paths(chosen)
