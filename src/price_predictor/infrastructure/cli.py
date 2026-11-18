"""CLI interface for the card price predictor.

The commands live in ``cli_train``, ``cli_predict``, ``cli_evaluate``,
``cli_serve`` and ``cli_corpus``, each with its parsers and handlers; this
module composes the parser and holds the run-manifest table.
"""

from __future__ import annotations

import argparse

from price_predictor.infrastructure.cli_corpus import (
    CHECK_REPORT_NAME,
    _check_report,
    _converted_root,
    add_corpus_commands,
    run_check_convert,
    run_convert,
    run_vocabulary,
)
from price_predictor.infrastructure.cli_evaluate import (
    add_evaluate,
    run_evaluate_sklearn,
    run_evaluate_transformer,
)
from price_predictor.infrastructure.cli_predict import (
    add_predict,
    run_predict_sklearn,
    run_predict_transformer,
)
from price_predictor.infrastructure.cli_run_manifest import READ_ONLY, ManifestTable, Writes
from price_predictor.infrastructure.cli_serve import add_serve, run_serve
from price_predictor.infrastructure.cli_train import (
    add_train,
    run_train_sklearn,
    run_train_transformer,
)

__all__ = [
    "CHECK_REPORT_NAME",
    "RUN_MANIFESTS",
    "build_parser",
    "run_check_convert",
    "run_convert",
    "run_evaluate_sklearn",
    "run_evaluate_transformer",
    "run_predict_sklearn",
    "run_predict_transformer",
    "run_serve",
    "run_train_sklearn",
    "run_train_transformer",
    "run_vocabulary",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="price_predictor",
        description="MTG card price predictor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    add_train(subparsers)
    add_predict(subparsers)
    add_evaluate(subparsers)
    add_serve(subparsers)
    add_corpus_commands(subparsers)
    return parser


# ── run manifests ─────────────────────────────────────────────────────


_DATASET_INPUTS = ("output_dir", "prices_path", "printings_path")

#: Each subcommand's run manifest (``.claude/skills/artifact-layout``): the
#: argument naming its output and the ones it reads.
RUN_MANIFESTS: ManifestTable = {
    "train sklearn": Writes("model_output", inputs=_DATASET_INPUTS),
    "train transformer": Writes("model_output", inputs=(*_DATASET_INPUTS, "vocab_path")),
    "predict sklearn": READ_ONLY,
    "predict transformer": READ_ONLY,
    "evaluate sklearn": Writes(
        "output_csv", file=True, inputs=("model_path", *_DATASET_INPUTS),
    ),
    "evaluate transformer": READ_ONLY,
    "serve": READ_ONLY,
    "vocabulary": Writes("output_dir", inputs=("cards_path", "printings_path")),
    "convert": Writes(_converted_root, inputs=("cards_path", "tokens_path")),
    # Its report lands in --report's directory, else in the converted tree it
    # checks, which it never creates.
    "check-convert": Writes(
        _check_report, file=True, create=lambda args: bool(args.report),
        inputs=("output_path", "cards_path"),
    ),
}
