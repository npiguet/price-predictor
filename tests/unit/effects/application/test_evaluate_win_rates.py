"""``--win-rates`` reaches the decodability battery (T108, Story 6 scenario 7)."""

from __future__ import annotations

from pathlib import Path

from effects.application.evaluate_effect_model import (
    DEFAULT_WIN_RATES,
    EvaluateEffectModelConfig,
    EvaluationReport,
)
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

