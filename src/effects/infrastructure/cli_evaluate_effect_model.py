"""``evaluate-effect-model``: the three gates and the reported checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from effects.application.evaluation_context import (
    BaselineVariantError,
    CorpusMismatchError,
)
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _EFFECT_MODEL_STORE,
    _LAYOUT_REFUSED,
    _MANIFEST_REFUSED,
    _SHARD_REFUSED,
    _SIDECARS_CHANGED,
    _add_cards_folder,
    _add_recorded_tokenizer_paths,
    _add_variant_scripts,
    _optional_path,
    _variant_scripts_help,
    config_from_args,
    exits_with,
)


def _evaluate_effect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate-effect-model",
        help="Run the three gates and the reported checks over a checkpoint",
    )
    parser.set_defaults(func=run_evaluate_effect_model)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="The checkpoint to evaluate",
    )
    parser.add_argument(
        "--variant-checkpoint", action="append", dest="baseline_checkpoints",
        type=_baseline_checkpoint, default=None, metavar="NAME=PATH",
        help="Baseline checkpoint to compare against; repeatable",
    )
    _add_cards_folder(parser, recorded=True)
    _add_variant_scripts(parser, help_text=_variant_scripts_help(
        "the tree the corpus's manifest records is read, if any"
    ))
    _add_recorded_tokenizer_paths(parser)
    parser.add_argument(
        "--corpus", type=_optional_path, default=None,
        help=(
            "A curated dataset directory (default: the one --checkpoint "
            "recorded); refused when rebuilt since training"
        ),
    )
    parser.add_argument(
        "--abilities-root", type=Path, required=True,
        help="The cache run encode-abilities wrote for this checkpoint",
    )
    parser.add_argument(
        "--win-rates", type=Path, required=True,
        help="Per-card win-rate table the decodability battery reads",
    )
    parser.add_argument(
        "--reports-dir", type=Path, required=True,
        help=(
            "Where the selection set frozen for the corpus is found, by the "
            "corpus digest, as on train-effect-model"
        ),
    )
    parser.add_argument(
        "--output", type=_optional_path, default=None,
        help=(
            "Directory the files the report names (the UMAP picture and its "
            "table) are written to (default: <checkpoint stem>.evaluation/ "
            "beside --checkpoint)"
        ),
    )


def _baseline_checkpoint(value: str):
    from effects.application.evaluation_context import parse_baseline_checkpoint

    try:
        return parse_baseline_checkpoint(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


@exits_with({
    f"{_EFFECT_MODEL_STORE}.HashMismatchError": 1,
    f"{_EFFECT_MODEL_STORE}.SplitMismatchError": 1,
    BaselineVariantError: 1,
    CorpusMismatchError: 1,
    _SIDECARS_CHANGED: 1,
    # A cache with no stamp, or stamped for another checkpoint (finding 11.7).
    "effects.domain.ability_cache_layout.CacheStampError": 1,
    FileNotFoundError: 1,
    # A checkpoint of another input layout, a corpus not wholly gen-3 (FR-076).
    _LAYOUT_REFUSED: 1,
    _MANIFEST_REFUSED: 1,
    UnacceptedGenerationError: 1,
    **_SHARD_REFUSED,
})
def run_evaluate_effect_model(args: argparse.Namespace) -> int:
    from effects.application.evaluate_effect_model import run as evaluate
    from effects.application.evaluation_context import EvaluateEffectModelConfig
    from effects.domain.evaluation_report import EvaluationReport

    # Each check prints as it is made, so a late failure leaves the gate
    # verdicts on the console and in the log; the closing line is the verdict.
    report = EvaluationReport(echo=lambda text: print(text, flush=True))
    evaluate(config_from_args(
        EvaluateEffectModelConfig, args,
        baseline_checkpoints=dict(args.baseline_checkpoints or []),
    ), report)
    print(report.verdict_line(), flush=True)
    # A failed gate is a result to record, not a reason to stop: the sweep runs
    # every arm's probes after evaluation whatever the verdicts.
    return 0
