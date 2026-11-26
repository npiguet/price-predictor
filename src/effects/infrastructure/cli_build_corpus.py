"""``build-corpus``: the raw shard corpus curated into a fixed dataset."""

from __future__ import annotations

import argparse

from effects.application.build_corpus_config import BuildCorpusConfig
from effects.domain.corpus_plan import BuildCorpusError
from effects.infrastructure.cli_common import (
    _MANIFEST_REFUSED,
    _add_cards_folder,
    _add_holdout_flags,
    _add_variant_scripts,
    _variant_scripts_help,
    add_config_flags,
    config_from_args,
    exits_with,
    holdout_rule,
)


def _class_mix(text: str) -> dict[str, float]:
    """``--class-mix`` parsed through the trainer's own ``parse_kind_mix``, so
    the two cannot disagree; a bad value lands in the usage message."""
    from effects.domain.sampling import parse_kind_mix

    try:
        return parse_kind_mix(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _build_corpus_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "build-corpus",
        help=(
            "Read the raw shard corpus once and write a fixed training "
            "corpus, two validation strata, and a manifest"
        ),
    )
    parser.set_defaults(func=run_build_corpus)
    _add_corpus_inputs(parser)
    _add_holdout_flags(parser)
    parser.add_argument(
        "--class-mix", type=_class_mix, default=None,
        help="class=share,… on-disk class proportions (default: the training mixture)",
    )
    add_config_flags(parser, BuildCorpusConfig, only=(
        "max_events_per_record", "game_disjoint_keywords",
        "text_cap", "reuse_cap", "training_records", "card_disjoint_text_cap",
        "validation_sample", "game_disjoint_share", "game_disjoint_keyword_share",
        "shard_records", "seed", "workers", "verify",
    ))


def _add_corpus_inputs(parser: argparse.ArgumentParser) -> None:
    add_config_flags(parser, BuildCorpusConfig, only=("records_dir", "output"))
    _add_cards_folder(parser)
    add_config_flags(parser, BuildCorpusConfig, only=("vocab_path",))
    # A string rather than a path, as --vocab-path: the manifest records it.
    _add_variant_scripts(parser, path_type=str, help_text=_variant_scripts_help(
        "a variant line resolves to no ability text, so it escapes the rarity "
        "table and the per-text cap"
    ))
    add_config_flags(parser, BuildCorpusConfig, only=("keyword_definitions",))


# A bare ValueError is NOT mapped: `decide`'s cap-mismatch guard and
# `CapHeap.merge`'s are internal invariants, worth a traceback.
@exits_with({
    BuildCorpusError: 1, FileNotFoundError: 1, _MANIFEST_REFUSED: 1,
    "price_predictor.domain.converted_provenance.ConversionRefused": 1,
    "effects.infrastructure.corpus_store.OutputsNotCleared": 1,
})
def run_build_corpus(args: argparse.Namespace) -> int:
    from effects.application import build_corpus as build_corpus_module
    from effects.infrastructure.corpus_conversion import require_collected_against

    config = config_from_args(BuildCorpusConfig, args, holdout=holdout_rule(args))
    require_collected_against(config.records_dir, config.cards_folders)
    return build_corpus_module.build(config)
