"""CLI interface for the effects module (``python -m effects``).

Each command's parser and ``run_*`` live in a ``cli_<command>.py`` beside this
module (the two collectors together in ``cli_collectors``); the config-flag
machinery, the refusal decorator and the flags several commands share are in
``cli_common``. This module composes the parser, holds the run-manifest table
and ``main``, and re-exports the names other code reaches through it.

Each ``run_*`` imports its application module when it runs, and the parser
imports nothing that loads torch, so ``--help`` never pays for it
(``tests/unit/effects/infrastructure/test_cli_parser_imports.py``). Every flag
and default is contract, pinned by ``specs/023-ability-effect-model/contracts/cli.md``:
a default that drifts changes what a corpus means, which is why a flag mirroring
a config field takes its default from the field (:func:`add_config_flags`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from effects.application.collect_variants import CollectVariantsConfig
from effects.application.collector_common import HOLDOUT_LIST_NAME
from effects.domain.baselines import Baseline
from effects.infrastructure.cli_build_corpus import _build_corpus_parser, run_build_corpus
from effects.infrastructure.cli_build_vocab import _build_vocab_parser, run_build_vocab
from effects.infrastructure.cli_collectors import (
    _collect_coverage_parser,
    _collect_variants_parser,
    collection_caps,
    collector_config,
    run_collect_coverage,
    run_collect_variants,
)
from effects.infrastructure.cli_common import (
    _exit_code,
    add_config_flags,
    config_from_args,
    exits_with,
    holdout_rule,
)
from effects.infrastructure.cli_encode_abilities import (
    _encode_abilities_parser,
    run_encode_abilities,
)
from effects.infrastructure.cli_evaluate_effect_model import (
    _evaluate_effect_model_parser,
    run_evaluate_effect_model,
)
from effects.infrastructure.cli_extract_keyword_definitions import (
    _extract_keyword_definitions_parser,
    run_extract_keyword_definitions,
)
from effects.infrastructure.cli_field_coverage import _field_coverage_parser, run_field_coverage
from effects.infrastructure.cli_holdout_cards import _holdout_cards_parser, run_holdout_cards
from effects.infrastructure.cli_rule_counts import _rule_counts_parser, run_rule_counts
from effects.infrastructure.cli_train_effect_model import (
    _train_effect_model_parser,
    run_train_effect_model,
)
from effects.infrastructure.cli_validate_corpus import (
    _validate_corpus_parser,
    run_validate_corpus,
    validation_ports,
)
from price_predictor.infrastructure.cli_run_manifest import (
    READ_ONLY,
    ManifestTable,
    Writes,
    dispatch,
    tag_subcommands,
)
from sealed.infrastructure.effect_collection_flags import candidate_rate_error, random_seat_error

#: Reached through this module by other code (``cli.exits_with``,
#: ``cli.config_from_args``, ``cli.validation_ports``, each ``run_*``).
__all__ = [
    "RUN_MANIFESTS",
    "_exit_code",
    "add_config_flags",
    "build_parser",
    "candidate_rate_error",
    "collection_caps",
    "collector_config",
    "config_from_args",
    "exits_with",
    "holdout_rule",
    "main",
    "random_seat_error",
    "run_build_corpus",
    "run_build_vocab",
    "run_collect_coverage",
    "run_collect_variants",
    "run_encode_abilities",
    "run_evaluate_effect_model",
    "run_extract_keyword_definitions",
    "run_field_coverage",
    "run_holdout_cards",
    "run_rule_counts",
    "run_train_effect_model",
    "run_validate_corpus",
    "validation_ports",
    "write_output_as_utf8",
]


# ── the table ───────────────────────────────────────────────────────────

_SUBCOMMAND_BUILDERS = (
    _build_vocab_parser,
    _extract_keyword_definitions_parser,
    _collect_coverage_parser,
    _build_corpus_parser,
    _field_coverage_parser,
    _rule_counts_parser,
    _holdout_cards_parser,
    _validate_corpus_parser,
    _collect_variants_parser,
    _train_effect_model_parser,
    _encode_abilities_parser,
    _evaluate_effect_model_parser,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="effects",
        description="Ability effect model: collection, training, encoding, evaluation",
    )
    subparsers = parser.add_subparsers(help="Available commands")
    for build in _SUBCOMMAND_BUILDERS:
        build(subparsers)
    return parser


def write_output_as_utf8(*streams) -> None:
    """Make each stream encode UTF-8, whatever the console's code page.

    Reports carry card texts and arrows, which a redirected stdout's cp1252 on
    Windows cannot encode. A stream that cannot be reconfigured, such as a
    test's ``StringIO``, is left as it is.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


# ── run manifests ───────────────────────────────────────────────────────


def _collection_dir(args: argparse.Namespace) -> Path | None:
    """A collector's shard directory, as ``collection_target`` resolves it;
    None where it names none, which the command refuses."""
    if args.training_corpus is not None:
        return Path(args.training_corpus)
    return Path(args.effect_records) if args.effect_records else None


def _collection_inputs(args: argparse.Namespace) -> list[Path | None]:
    """The lists and the checkpoint a collector reads its holdout from."""
    corpus_list = (
        Path(args.training_corpus) / HOLDOUT_LIST_NAME
        if args.training_corpus is not None else None
    )
    return [
        args.exclude_cards, args.split_from, corpus_list, getattr(args, "only_cards", None),
        getattr(args, "wording_cards", None), *(getattr(args, "count_from", None) or ()),
    ]


def _steered_catalogue(args: argparse.Namespace) -> Path | None:
    """The rules catalogue a steered variant round reads; None for the untargeted."""
    from effects.domain.variant_rounds import VariantRound

    return CollectVariantsConfig.rules_dir if args.round is VariantRound.STEERED else None


def _evaluation_dir(args: argparse.Namespace) -> Path:
    """Where the evaluation's pictures go: ``--output``, else
    ``<checkpoint stem>.evaluation/`` beside the checkpoint."""
    from effects.application.evaluation_context import evaluation_dir

    return evaluation_dir(args.checkpoint, args.output)


def _corpus_variant_tree(args: argparse.Namespace) -> list[Path]:
    """The variant tree ``--variant-scripts`` names, else the one the corpus
    manifest records: what a trainer or an evaluation reads without a flag."""
    from effects.infrastructure.recorded_inputs import corpus_variant_tree

    return corpus_variant_tree(args.corpus, args.variant_scripts)


def _evaluated_inputs(args: argparse.Namespace) -> list[Path]:
    """What ``evaluate-effect-model`` reads from its checkpoint: the
    vocabulary, the keyword table, the corpus and its variant tree."""
    from effects.infrastructure.recorded_inputs import corpus_paths, tokenizer_paths

    return [
        *tokenizer_paths(args.checkpoint, vocab=args.vocab_path, keywords=args.keyword_definitions),
        *corpus_paths(args.checkpoint, corpus=args.corpus, variant_scripts=args.variant_scripts),
    ]


def _evaluated_trees(args: argparse.Namespace) -> list[Path]:
    """The converted trees ``evaluate-effect-model`` reads: ``--cards-folder``,
    else the folders the checkpoint recorded."""
    from effects.infrastructure.recorded_inputs import checkpoint_cards_folders

    return checkpoint_cards_folders(args.checkpoint, args.cards_folders)


def _cache_stamps(args: argparse.Namespace) -> list[Path]:
    """The stamp of each variant's cache the evaluation reads."""
    from effects.domain.ability_cache_layout import stamp_path

    variants = (Baseline.FULL, Baseline.NO_STATE, Baseline.TAXONOMY)
    stamps = (stamp_path(args.abilities_root, variant) for variant in variants)
    return [stamp for stamp in stamps if stamp.is_file()]


def _encoded_inputs(args: argparse.Namespace) -> list[Path]:
    """What ``encode-abilities`` reads from its checkpoint: the vocabulary and
    the keyword table."""
    from effects.infrastructure.recorded_inputs import tokenizer_paths

    return tokenizer_paths(
        args.checkpoint, vocab=args.vocab_path, keywords=args.keyword_definitions,
    )


_SOURCE_TREES = ("cards_folders", "variant_scripts")
_TOKENIZER_PATHS = ("vocab_path", "keyword_definitions")

#: Each subcommand's run manifest (``.claude/skills/artifact-layout``): the
#: argument naming its output and the ones it reads.
RUN_MANIFESTS: ManifestTable = {
    "build-vocab": Writes(
        "vocab_path", file=True, inputs=("cards_folders", "keyword_definitions"),
    ),
    "extract-keyword-definitions": Writes("output", file=True),
    "collect-coverage": Writes(
        _collection_dir, inputs=("cards_folders", _collection_inputs),
    ),
    "build-corpus": Writes(
        "output", when=lambda args: not args.verify,
        inputs=("records_dir", *_SOURCE_TREES, *_TOKENIZER_PATHS),
    ),
    "field-coverage": READ_ONLY,
    "rule-counts": Writes(
        "output", file=True, inputs=("records_dir", *_SOURCE_TREES, "rules_dir"),
    ),
    "holdout-cards": Writes("out", file=True, inputs=("cards_folders", "printings_path")),
    "validate-corpus": READ_ONLY,
    # It reads and extends --variant-scripts, an inputs-kind tree shared by
    # every variant run (a variant is reused by its digest).
    "collect-variants": Writes(
        _collection_dir,
        inputs=(
            "forge_cards_path", "variant_scripts", "rule_counts", _steered_catalogue,
            "cards_folders", _collection_inputs,
        ),
    ),
    "train-effect-model": Writes(
        "model_output",
        inputs=(
            "corpus", *_SOURCE_TREES, _corpus_variant_tree, *_TOKENIZER_PATHS, "reports_dir",
        ),
    ),
    "encode-abilities": Writes(
        "output_root", inputs=("checkpoint", *_SOURCE_TREES, _encoded_inputs),
    ),
    "evaluate-effect-model": Writes(
        _evaluation_dir,
        inputs=(
            "checkpoint",
            lambda args: [path for _name, path in (args.baseline_checkpoints or [])],
            _evaluated_trees, "variant_scripts", _evaluated_inputs, "abilities_root",
            _cache_stamps, "win_rates", "reports_dir",
        ),
    ),
}


def main() -> None:
    write_output_as_utf8(sys.stdout, sys.stderr)
    # Stamped, because these commands run for hours: a progress line without a
    # clock answers "is it alive" and not "when did it last move".
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    parser = build_parser()
    tag_subcommands(parser)
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(dispatch(args, RUN_MANIFESTS, module="effects"))
