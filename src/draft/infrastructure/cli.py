"""The draft CLI's entry point: the parser, the run manifests and ``main``.

Each command's parser and runner live in a ``cli_<command>.py`` module beside
this one, shared flags in ``cli_common.py``. ``build_parser`` adds one subparser
per command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and
calls the resolved ``func``. Application imports are lazy (inside each ``run_*``)
so ``python -m draft --help`` and unit tests don't pay the torch import cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft.infrastructure.cli_analyze_generated_decks import _build_analyze_generated_decks_parser
from draft.infrastructure.cli_generate_draft_data import _build_generate_draft_data_parser
from draft.infrastructure.cli_play_draft_games import _build_play_draft_games_parser
from draft.infrastructure.cli_train_draft_agent import _build_train_draft_agent_parser
from draft.infrastructure.cli_train_draft_agent_online import (
    _build_train_draft_agent_online_parser,
)
from draft.infrastructure.cli_train_draft_agent_rl import _build_train_draft_agent_rl_parser
from draft.infrastructure.cli_validate_builder import _build_validate_builder_parser
from price_predictor.infrastructure.cli_run_manifest import (
    READ_ONLY,
    ManifestTable,
    Writes,
    dispatch,
    labelled_paths,
    tag_subcommands,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draft",
        description="Draft agent data generation and training tools",
    )
    subparsers = parser.add_subparsers(help="Available commands")
    _build_generate_draft_data_parser(subparsers)
    _build_train_draft_agent_parser(subparsers)
    _build_train_draft_agent_rl_parser(subparsers)
    _build_train_draft_agent_online_parser(subparsers)
    _build_validate_builder_parser(subparsers)
    _build_analyze_generated_decks_parser(subparsers)
    _build_play_draft_games_parser(subparsers)
    return parser


# ── run manifests ─────────────────────────────────────────────────────

def _offline_checkpoint_dir(args: argparse.Namespace) -> Path | None:
    """An offline trainer's run directory, resolved as the trainer resolves it:
    ``--checkpoint-dir``, else on ``--resume`` the directory the resumed run
    recorded; ``None`` (no manifest) when neither names one, which the trainer refuses."""
    if args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir)
    if args.resume is not None:
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        try:
            recorded = (DraftAgentStore().load_checkpoint(Path(args.resume)).train_config
                        or {}).get("checkpoint_dir")
        except (OSError, ValueError, KeyError, RuntimeError):
            recorded = None  # the trainer reports an unreadable checkpoint itself
        if recorded:
            return Path(recorded)
    return None


def _offline_training_inputs(args: argparse.Namespace) -> list[str | Path | None]:
    """What an offline trainer reads; a resume reads its recorded paths unless overridden."""
    def path(name: str) -> str | Path | None:
        given = getattr(args, name)
        return given

    return [
        path("drafts_path"), path("cards_path"), args.embeddings_path,
        args.resume, args.checkpoint,
    ]


#: Each subcommand's run manifest (``.claude/skills/artifact-layout``): the
#: argument naming its output and the ones it reads.
RUN_MANIFESTS: ManifestTable = {
    "generate-draft-data": Writes(
        "output_path", file=True,
        inputs=(
            "scorer_checkpoint", "picker_checkpoint", "cards_path", "embeddings_path",
            lambda args: labelled_paths(args.agent_checkpoints),
        ),
    ),
    "train-draft-agent": Writes(
        _offline_checkpoint_dir, inputs=(_offline_training_inputs,),
    ),
    "train-draft-agent-rl": Writes(
        _offline_checkpoint_dir,
        inputs=(_offline_training_inputs, "critic_corpus"),
    ),
    # Its run directory is --checkpoint-dir.
    "train-draft-agent-online": Writes(
        "checkpoint_dir",
        inputs=(
            lambda args: labelled_paths(args.learner),
            lambda args: labelled_paths(args.frozen),
            "scorer_checkpoint", "picker_checkpoint", "cards_path", "embeddings_path",
        ),
    ),
    "validate-builder": READ_ONLY,
    "analyze-generated-decks": READ_ONLY,
    "play-draft-games": Writes("output_path", file=True, inputs=("drafts_path",)),
}


def main() -> None:
    parser = build_parser()
    tag_subcommands(parser)
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(dispatch(args, RUN_MANIFESTS, module="draft"))
