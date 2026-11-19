"""Configs whose paths a run names, built with placeholders for the ones a test does not read.

``EvaluateEffectModelConfig`` has no default for the checkpoint, the cache
root, the win-rate table or the reports directory (artifact-layout rule 2),
so a test that reads none of them names an empty one here.
"""

from __future__ import annotations

from pathlib import Path

from effects.application.evaluation_context import EvaluateEffectModelConfig


def evaluation_config(**options) -> EvaluateEffectModelConfig:
    """An evaluation config; each path ``options`` does not give is a placeholder."""
    return EvaluateEffectModelConfig(**{
        "checkpoint": Path("latest.pt"), "abilities_root": Path("cache"),
        "win_rates": Path("rates.txt"), "reports_dir": Path("reports"), **options,
    })

