"""The sealed CLI's entry point: the parser, the run manifests and ``main``.

Each command's parser and runner live in a ``cli_<command>.py`` module beside
this one."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.cli_run_manifest import (
    READ_ONLY,
    ManifestTable,
    Writes,
    dispatch,
    tag_subcommands,
)
from sealed.infrastructure.cli_build_vocab import _build_build_vocab_parser
from sealed.infrastructure.cli_decks import (
    _build_analyze_generated_decks_parser,
    _build_build_decks_parser,
    _build_pick_decks_parser,
)
from sealed.infrastructure.cli_encode_cards import _build_encode_cards_parser
from sealed.infrastructure.cli_evaluate_scorer import _build_evaluate_scorer_parser
from sealed.infrastructure.cli_generate_pools import _build_generate_pools_parser, _pools_dir
from sealed.infrastructure.cli_match_outcomes import _build_match_outcomes_parser, outcomes_dir
from sealed.infrastructure.cli_train_encoder import _build_train_encoder_parser
from sealed.infrastructure.cli_train_picker import _build_train_picker_parser
from sealed.infrastructure.cli_train_scorer import _build_train_scorer_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sealed",
        description="Sealed dataset preparation tools",
    )
    subparsers = parser.add_subparsers(help="Available commands")

    _build_encode_cards_parser(subparsers)
    _build_generate_pools_parser(subparsers)
    _build_build_decks_parser(subparsers)
    _build_train_scorer_parser(subparsers)
    _build_evaluate_scorer_parser(subparsers)
    _build_match_outcomes_parser(subparsers)
    _build_train_encoder_parser(subparsers)
    _build_build_vocab_parser(subparsers)
    _build_train_picker_parser(subparsers)
    _build_pick_decks_parser(subparsers)
    _build_analyze_generated_decks_parser(subparsers)

    return parser


# ── run manifests ─────────────────────────────────────────────────────


def _checkpoint_dir(args: argparse.Namespace) -> Path | None:
    """A trainer's checkpoint directory: ``--checkpoint-dir``, else the one
    ``--resume``'s checkpoint sits in (where the resumed run writes), else
    None (the run is refused)."""
    if args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir)
    return Path(args.resume).parent if args.resume is not None else None


def _train_scorer_inputs(args: argparse.Namespace) -> list[str | None]:
    """What a scorer run reads; a resume reads its recorded paths unless overridden."""
    return [
        args.outcomes_path, args.cards_path,
        args.embeddings_path, args.resume, args.scorer_checkpoint, args.encoder_checkpoint,
    ]


def _train_picker_inputs(args: argparse.Namespace) -> list[str | Path | None]:
    """What a picker run reads; a resume reads its recorded paths unless overridden."""
    return [
        args.pools_path, args.scorer_checkpoint, args.auditor_scorer_checkpoint,
        args.cards_path, args.embeddings_path, args.resume, args.picker_checkpoint,
    ]


#: Each subcommand's run manifest (``.claude/skills/artifact-layout``): the
#: argument naming its output and the ones it reads.
RUN_MANIFESTS: ManifestTable = {
    # The embeddings land in --embeddings-path, created for them, else beside
    # the cards, in a tree it never creates.
    "encode-cards": Writes(
        lambda args: args.embeddings_path or args.cards_path,
        create=lambda args: args.embeddings_path is not None,
        inputs=("encoder_checkpoint", "scorer_checkpoint", "vocab_path", "cards_path"),
    ),
    "generate-pools": Writes(_pools_dir, inputs=("exclude_cards",)),
    "build-decks": Writes(
        "output", file=True,
        inputs=("pools_path", "checkpoint", "cards_path", "embeddings_path"),
    ),
    "train-scorer": Writes(_checkpoint_dir, inputs=(_train_scorer_inputs,)),
    # Without --work-dir the evaluation works in a temporary directory.
    "evaluate-scorer": Writes(
        "work_dir", inputs=("checkpoint", "cards_path", "embeddings_path"),
    ),
    # A collection run's directory is its --effect-records, which also holds
    # the two sealed corpora unless --outcomes-dir sends them elsewhere.
    "match-outcomes": Writes(
        lambda args: args.effect_records or outcomes_dir(args),
        inputs=("side_a_decks", "side_b_decks", "exclude_cards", "cards_folder"),
    ),
    "train-encoder": Writes(
        "model_output", inputs=("cards_played_path", "cards_folder", "vocab_path"),
    ),
    "build-vocab": Writes("vocab_path", file=True, inputs=("cards_folder",)),
    "train-picker": Writes(_checkpoint_dir, inputs=(_train_picker_inputs,)),
    "pick-decks": Writes(
        "output", file=True,
        inputs=("pools_path", "picker_checkpoint", "cards_path", "embeddings_path"),
    ),
    "analyze-generated-decks": READ_ONLY,
}


def main() -> None:
    parser = build_parser()
    tag_subcommands(parser)
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(dispatch(args, RUN_MANIFESTS, module="sealed"))
