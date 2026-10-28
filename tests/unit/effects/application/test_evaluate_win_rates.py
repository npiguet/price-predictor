"""Where the evaluator's paths come from.

``--win-rates`` reaches the decodability battery (T108, Story 6 scenario 7),
and the variant tree defaults to the one the checkpoint's corpus recorded.
"""

from __future__ import annotations

from pathlib import Path

from effects.application.evaluation_context import (
    DEFAULT_WIN_RATES,
    EvaluateEffectModelConfig,
)
from effects.domain.evaluation_report import EvaluationReport
from effects.infrastructure.cli import build_parser


def _config_from(argv, monkeypatch) -> EvaluateEffectModelConfig:
    seen = {}

    def fake_run(config):
        seen["config"] = config
        return EvaluationReport()

    monkeypatch.setattr("effects.application.evaluate_effect_model.run", fake_run)
    args = build_parser().parse_args(["evaluate-effect-model", *argv])
    assert args.func(args) == 0
    return seen["config"]


def test_the_flag_names_the_table_the_battery_reads(monkeypatch, tmp_path):
    table = tmp_path / "rates.txt"
    config = _config_from(["--win-rates", str(table)], monkeypatch)
    assert config.win_rates == table


def test_the_default_is_the_sealed_pipeline_s_table(monkeypatch):
    config = _config_from([], monkeypatch)
    assert config.win_rates == DEFAULT_WIN_RATES == Path(
        "output/sealed/cards-win-rates.txt",
    )
    assert EvaluateEffectModelConfig().win_rates == DEFAULT_WIN_RATES



# ── defaults the corpus manifest supplies ────────────────────────────────


def test_the_variant_tree_defaults_to_the_one_the_corpus_recorded():
    """Without it every variant card's ability key resolves to no text."""
    from types import SimpleNamespace

    from effects.application.evaluation_context import with_corpus_defaults

    manifest = SimpleNamespace(variant_scripts="output/effects/variant-scripts/")
    config = with_corpus_defaults(EvaluateEffectModelConfig(), manifest)
    assert config.variant_scripts == Path("output/effects/variant-scripts/")


def test_an_explicit_variant_tree_or_none_recorded_is_kept():
    from types import SimpleNamespace

    from effects.application.evaluation_context import with_corpus_defaults

    chosen = EvaluateEffectModelConfig(variant_scripts=Path("elsewhere"))
    recorded = SimpleNamespace(variant_scripts="output/effects/variant-scripts/")
    assert with_corpus_defaults(chosen, recorded).variant_scripts == Path("elsewhere")
    blank = SimpleNamespace(variant_scripts="")
    assert with_corpus_defaults(EvaluateEffectModelConfig(), blank).variant_scripts is None
