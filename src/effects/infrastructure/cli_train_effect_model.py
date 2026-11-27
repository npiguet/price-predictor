"""``train-effect-model``: the encoder and effect head trained from a curated corpus."""

from __future__ import annotations

import argparse
import logging

from effects.application.closed_list_check import ClosedListMismatchError
from effects.application.train_effect_model import (
    EmptyHoldoutError,
    HoldoutMismatchError,
    MissingTrainingInputError,
    SurfaceMismatchError,
    TrainEffectModelConfig,
    ValidationGameInTrainingError,
)
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.infrastructure.cli_common import (
    _EFFECT_MODEL_STORE,
    _LAYOUT_REFUSED,
    _MANIFEST_REFUSED,
    _SHARD_REFUSED,
    _SIDECARS_CHANGED,
    _add_cards_folder,
    _add_variant,
    _add_variant_scripts,
    _variant_scripts_help,
    add_config_flags,
    config_from_args,
    exits_with,
)

logger = logging.getLogger(__name__)


def _train_effect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-effect-model",
        help="Train the ability encoder and effect head jointly from random init",
    )
    parser.set_defaults(func=run_train_effect_model)
    parser.add_argument(
        "--corpus", type=str, required=True,
        help=(
            "A curated dataset directory from build-corpus, the trainer's only "
            "input: its manifest records the split, the samples and the rarity table"
        ),
    )
    _add_cards_folder(parser)
    _add_variant_scripts(parser, help_text=_variant_scripts_help(
        "the tree the corpus's manifest records is read, if any"
    ))
    _add_variant(parser, help_text=(
        "The shipping model, or one of its baselines, which zero part of the "
        "input or replace the encoder; each trains under its own name "
        "(default: %(default)s)"
    ))
    add_config_flags(parser, TrainEffectModelConfig)


@exits_with({
    MissingTrainingInputError: 1,
    _LAYOUT_REFUSED: 1,
    _MANIFEST_REFUSED: 1,
    _SIDECARS_CHANGED: 1,
    HoldoutMismatchError: 1,
    ClosedListMismatchError: 1,
    # Raised in a prefetch worker too, which hands the exception itself over.
    ValidationGameInTrainingError: 1,
    UnacceptedGenerationError: 1,
    **_SHARD_REFUSED,
    SurfaceMismatchError: 1,
    # A --keyword-definitions that names no file, or not the corpus's.
    "effects.application.tokenizer_inputs.KeywordDefinitionsError": 1,
    # Shards under the corpus that its manifest does not list.
    "effects.infrastructure.curated_shards.CorpusShardsError": 1,
    EmptyHoldoutError: 1,
    # A resume under other settings (contracts/cli.md).
    f"{_EFFECT_MODEL_STORE}.ResumeMismatchError": 1,
    # No selection set frozen for the corpus, or one frozen for another or
    # edited (research E1); a control-score component with nothing to score (FR-068c).
    "effects.domain.selection_set.SelectionSetError": 1,
    "effects.application.control_items.ControlScoreError": 1,
    # A missing manifest or resume file.
    FileNotFoundError: 1,
})
def run_train_effect_model(args: argparse.Namespace) -> int:
    from effects.application.train_effect_model import run as train

    try:
        config = config_from_args(TrainEffectModelConfig, args)
    except ValueError as exc:
        # An encoder shape the config refuses; a ValueError out of training
        # itself is an invariant and keeps its traceback.
        logger.error("%s", exc)
        return 1
    return train(config)
