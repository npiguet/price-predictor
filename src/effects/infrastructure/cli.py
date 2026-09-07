"""CLI interface for the effects module (``python -m effects``).

Mirrors ``draft/infrastructure/cli.py``: ``build_parser`` adds one subparser per
command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and calls the
resolved ``func``. Application imports are lazy (inside each ``run_*``) so
``python -m effects --help`` and the unit suite don't pay the torch import cost.

Every flag and default here is contract, pinned by
``specs/023-ability-effect-model/contracts/cli.md`` and asserted by the
contract tests — a default that drifts changes what a corpus means.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── shared defaults (contracts/cli.md) ──────────────────────────────────
DEFAULT_RECORDS_DIR = "output/effects/records/"
DEFAULT_CARDS_FOLDERS = ("output/cardsfolder/", "output/tokenscripts/")
DEFAULT_KEYWORD_DEFINITIONS = "output/effects/keyword-definitions.json"
DEFAULT_VOCAB_PATH = "models/effects/vocab.txt"
DEFAULT_SCRIPT_VOCAB_PATH = "models/effects/vocab-script.txt"
DEFAULT_VARIANT_SCRIPTS = "output/effects/variant-scripts/"
DEFAULT_ABILITIES_ROOT = "output/effects/abilities/"
DEFAULT_CHECKPOINT = "models/effects/effect-model/latest.pt"
DEFAULT_PRINTINGS = "resources/AllPrintings.json"

#: Caps and budgets, shared by every collecting supervisor (FR-028).
CAP_DEFAULTS: dict[str, object] = {
    "mana_cap": 2000,
    "playability_rate": 0.1,
    "interventions_per_game": 2,
    "probes_per_game": 2,
    "probe_keywords": "",
}


def _add_cards_folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-folder", action="append", dest="cards_folders", default=None,
        help=(
            "Converted source tree; repeatable. Defaults to "
            f"{' and '.join(DEFAULT_CARDS_FOLDERS)}."
        ),
    )


def _add_cap_flags(parser: argparse.ArgumentParser) -> None:
    """The cap and budget flags every collecting supervisor carries."""
    parser.add_argument(
        "--mana-cap", type=int, default=CAP_DEFAULTS["mana_cap"],
        help=(
            "Resolution records per unique mana-ability text, per worker "
            "process (default: 2000)"
        ),
    )
    parser.add_argument(
        "--playability-rate", type=float,
        default=CAP_DEFAULTS["playability_rate"],
        help=(
            "Fraction of decision-subkind logging points sampled; attackers "
            "and blockers are always logged (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--interventions-per-game", type=int,
        default=CAP_DEFAULTS["interventions_per_game"],
        help="Interventional resolutions per game, stage three (default: 2)",
    )
    parser.add_argument(
        "--probes-per-game", type=int, default=CAP_DEFAULTS["probes_per_game"],
        help="Damage-step probe forks per game (default: 2)",
    )
    parser.add_argument(
        "--probe-keywords", type=str, default=CAP_DEFAULTS["probe_keywords"],
        help=(
            "Comma-separated canary-failing keywords to probe. Empty (the "
            "default) takes no probe fork at all, whatever the build state."
        ),
    )


def resolve_cards_folders(values: list[str] | None) -> tuple[Path, ...]:
    """``--cards-folder`` values, or both converted trees."""
    chosen = values or list(DEFAULT_CARDS_FOLDERS)
    return tuple(Path(value) for value in chosen)


# ── build-vocab ─────────────────────────────────────────────────────────


def _build_vocab_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "build-vocab",
        help="Build the effects tokenizer vocabulary from the converted corpus",
    )
    parser.set_defaults(func=run_build_vocab)
    parser.add_argument(
        "--surface", choices=("prose", "script"), default="prose",
        help=(
            "prose (default) reads the converted text; script adds the "
            "sidecars' script lines and switches the --vocab-path default"
        ),
    )
    _add_cards_folder(parser)
    parser.add_argument(
        "--vocab-path", type=str, default=None,
        help=(
            f"Output path (default: {DEFAULT_VOCAB_PATH}, or "
            f"{DEFAULT_SCRIPT_VOCAB_PATH} under --surface script)"
        ),
    )
    parser.add_argument(
        "--keyword-definitions", type=str, default=DEFAULT_KEYWORD_DEFINITIONS,
        help=f"Keyword definitions to scan (default: {DEFAULT_KEYWORD_DEFINITIONS})",
    )
    parser.add_argument(
        "--target-size", type=int, default=5000,
        help="Post-truncate the corpus-frequency vocabulary (default: 5000)",
    )
    parser.add_argument(
        "--printings-path", type=str, default=DEFAULT_PRINTINGS,
        help=f"MTGJSON dump for set-code seeding (default: {DEFAULT_PRINTINGS})",
    )


def run_build_vocab(args: argparse.Namespace) -> int:
    from effects.application.build_vocab import (
        BuildVocabConfig,
        EmptyCardsFolderError,
    )
    from effects.application.build_vocab import run as build_vocab

    config = BuildVocabConfig(
        surface=args.surface,
        cards_folders=resolve_cards_folders(args.cards_folders),
        vocab_path=Path(args.vocab_path) if args.vocab_path else None,
        keyword_definitions=Path(args.keyword_definitions),
        target_size=args.target_size,
        printings_path=Path(args.printings_path),
    )
    try:
        build_vocab(config)
    except EmptyCardsFolderError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    return 0


# ── extract-keyword-definitions ─────────────────────────────────────────


def _extract_keyword_definitions_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract-keyword-definitions",
        help="Write Forge's keyword table (reminder-text templates) as JSON",
    )
    parser.set_defaults(func=run_extract_keyword_definitions)
    parser.add_argument(
        "--output", type=str, default=DEFAULT_KEYWORD_DEFINITIONS,
        help=f"Output path (default: {DEFAULT_KEYWORD_DEFINITIONS})",
    )


def run_extract_keyword_definitions(args: argparse.Namespace) -> int:
    from effects.application.extract_keyword_definitions import (
        ExtractKeywordDefinitionsConfig,
    )
    from effects.application.extract_keyword_definitions import (
        run as extract,
    )

    return extract(ExtractKeywordDefinitionsConfig(output=Path(args.output)))


# ── collect-coverage ────────────────────────────────────────────────────


def _collect_coverage_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "collect-coverage",
        help=(
            "Play weighted decks over the whole converted corpus until every "
            "card is satisfied or retired"
        ),
    )
    parser.set_defaults(func=run_collect_coverage)
    parser.add_argument(
        "--effect-records", type=str, default=DEFAULT_RECORDS_DIR,
        help=(
            "Destination shard directory. Unlike on match-outcomes this has a "
            f"default ({DEFAULT_RECORDS_DIR}): collection is the whole point "
            "of this command."
        ),
    )
    _add_cards_folder(parser)
    parser.add_argument(
        "--split-from", type=str, default=None,
        help=(
            "A checkpoint whose held-out cards are excluded from every deck. "
            "Without it a coverage run would contaminate the card-disjoint "
            "split of the model it feeds."
        ),
    )
    parser.add_argument(
        "--target-records", type=int, default=50,
        help="Per-card satisfaction goal (default: 50)",
    )
    parser.add_argument(
        "--decks-per-round", type=int, default=500,
        help="Decks played as matches per round (default: 500)",
    )
    parser.add_argument(
        "--no-progress-rounds", type=int, default=3,
        help=(
            "Consecutive rounds without a new qualifying record before a card "
            "retires (default: 3). This is what makes the run terminate."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=12,
        help="Parallel Java worker processes (default: 12)",
    )
    _add_cap_flags(parser)


def run_collect_coverage(args: argparse.Namespace) -> int:
    from effects.application.collect_coverage import CollectCoverageConfig
    from effects.application.collect_coverage import run as collect

    config = CollectCoverageConfig(
        effect_records=Path(args.effect_records),
        cards_folders=resolve_cards_folders(args.cards_folders),
        split_from=Path(args.split_from) if args.split_from else None,
        target_records=args.target_records,
        decks_per_round=args.decks_per_round,
        no_progress_rounds=args.no_progress_rounds,
        workers=args.workers,
    )
    return collect(config)


# ── train-effect-model ──────────────────────────────────────────────────


def _train_effect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-effect-model",
        help="Train the ability encoder and effect head jointly from random init",
    )
    parser.set_defaults(func=run_train_effect_model)
    parser.add_argument("--records-dir", type=str, default=DEFAULT_RECORDS_DIR)
    _add_cards_folder(parser)
    parser.add_argument(
        "--variant-scripts", type=str, default=None,
        help=f"Stage four: perturbed scripts (default: {DEFAULT_VARIANT_SCRIPTS})",
    )
    parser.add_argument(
        "--split-from", type=str, default=None,
        help=(
            "Inherit another checkpoint's split, vocabulary and "
            "keyword-definition paths. Required for every --variant run."
        ),
    )
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--printings-path", type=str, default=DEFAULT_PRINTINGS)
    parser.add_argument(
        "--keyword-definitions", type=str, default=DEFAULT_KEYWORD_DEFINITIONS,
    )
    parser.add_argument(
        "--model-output", type=str, default=None,
        help=(
            "Checkpoint directory (default: models/effects/effect-model/ for "
            "--variant full, models/effects/effect-model/{variant}/ otherwise)"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("full", "identity", "state-only", "no-state", "taxonomy"),
        default="full",
    )
    parser.add_argument("--e-dim", type=int, default=64)
    parser.add_argument("--e-noise", type=float, default=0.05)
    parser.add_argument("--keyword-expand-p", type=float, default=0.25)
    parser.add_argument("--context-dropout", type=float, default=0.15)
    parser.add_argument("--mlm-weight", type=float, default=0.1)
    parser.add_argument("--mlm-mask-prob", type=float, default=0.15)
    parser.add_argument("--api-weight", type=float, default=0.05)
    parser.add_argument("--curriculum-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--kind-mix", type=str, default=None,
        help="class=share,… (default: the eight-class mixture)",
    )
    parser.add_argument(
        "--context-cache", action="store_true",
        help=(
            "Switch context gradient to a stop-gradient momentum cache — the "
            "documented fallback when live re-encoding exceeds the GPU budget"
        ),
    )
    parser.add_argument("--cache-refresh", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--withhold-keyword", type=str, default=None,
        help=(
            "Hold one implemented keyword's token out of training so the "
            "zero-shot check has something to measure; its occurrences are "
            "always expanded instead"
        ),
    )


def run_train_effect_model(args: argparse.Namespace) -> int:
    from effects.application.train_effect_model import (
        MissingSplitError,
        TrainEffectModelConfig,
        require_split_from,
    )

    split_from = Path(args.split_from) if args.split_from else None
    try:
        require_split_from(args.variant, split_from)
    except MissingSplitError as exc:
        logger.error("%s", exc)
        return 2

    config = TrainEffectModelConfig(
        records_dir=Path(args.records_dir),
        cards_folders=resolve_cards_folders(args.cards_folders),
        variant_scripts=(
            Path(args.variant_scripts) if args.variant_scripts else None
        ),
        split_from=split_from,
        vocab_path=Path(args.vocab_path),
        printings_path=Path(args.printings_path),
        keyword_definitions=Path(args.keyword_definitions),
        model_output=Path(args.model_output) if args.model_output else None,
        variant=args.variant,
        e_dim=args.e_dim,
        e_noise=args.e_noise,
        keyword_expand_p=args.keyword_expand_p,
        context_dropout=args.context_dropout,
        mlm_weight=args.mlm_weight,
        mlm_mask_prob=args.mlm_mask_prob,
        api_weight=args.api_weight,
        curriculum_step=args.curriculum_step,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        kind_mix=args.kind_mix,
        context_cache=args.context_cache,
        cache_refresh=args.cache_refresh,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        patience=args.patience,
        withhold_keyword=args.withhold_keyword,
    )
    from effects.application.train_effect_model import run as train

    return train(config)


# ── encode-abilities ────────────────────────────────────────────────────


def _encode_abilities_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "encode-abilities",
        help="Compute the per-ability embedding cache from a trained checkpoint",
    )
    parser.set_defaults(func=run_encode_abilities)
    parser.add_argument(
        "--variant",
        choices=("full", "identity", "state-only", "no-state", "taxonomy"),
        default="full",
        help="Resolves the checkpoint and the output suffix together",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Overrides the checkpoint --variant resolves",
    )
    _add_cards_folder(parser)
    parser.add_argument("--variant-scripts", type=str, default=None)
    parser.add_argument(
        "--vocab-path", type=str, default=None,
        help="Defaults to the path the checkpoint recorded from training",
    )
    parser.add_argument(
        "--keyword-definitions", type=str, default=None,
        help="Defaults to the path the checkpoint recorded from training",
    )
    parser.add_argument("--output-root", type=str, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove only the files this variant wrote, then re-encode",
    )


def run_encode_abilities(args: argparse.Namespace) -> int:
    from effects.application.encode_abilities import EncodeAbilitiesConfig
    from effects.application.encode_abilities import run as encode
    from effects.infrastructure.effect_model_store import HashMismatchError

    config = EncodeAbilitiesConfig(
        variant=args.variant,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        cards_folders=resolve_cards_folders(args.cards_folders),
        variant_scripts=(
            Path(args.variant_scripts) if args.variant_scripts else None
        ),
        vocab_path=Path(args.vocab_path) if args.vocab_path else None,
        keyword_definitions=(
            Path(args.keyword_definitions) if args.keyword_definitions else None
        ),
        output_root=Path(args.output_root),
        clean=args.clean,
    )
    try:
        encode(config)
    except HashMismatchError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    return 0


# ── evaluate-effect-model ───────────────────────────────────────────────


def _evaluate_effect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate-effect-model",
        help="Run the three gates and the reported checks over a checkpoint",
    )
    parser.set_defaults(func=run_evaluate_effect_model)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--variant-checkpoint", action="append", dest="variant_checkpoints",
        default=None, metavar="NAME=PATH",
        help="Baseline checkpoint to compare against; repeatable",
    )
    parser.add_argument("--records-dir", type=str, default=DEFAULT_RECORDS_DIR)
    _add_cards_folder(parser)
    parser.add_argument("--variant-scripts", type=str, default=None)
    parser.add_argument(
        "--vocab-path", type=str, default=None,
        help="Defaults to the path --checkpoint recorded",
    )
    parser.add_argument(
        "--keyword-definitions", type=str, default=None,
        help="Defaults to the path --checkpoint recorded",
    )
    parser.add_argument("--abilities-root", type=str, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--sealed-encoder-checkpoint", type=str,
        default="models/sealed/encoder/latest.pt",
        help="Read side by side with the effects cache in the decodability battery",
    )


def run_evaluate_effect_model(args: argparse.Namespace) -> int:
    from effects.application.evaluate_effect_model import (
        EvaluateEffectModelConfig,
        parse_variant_checkpoint,
    )
    from effects.application.evaluate_effect_model import run as evaluate
    from effects.infrastructure.effect_model_store import (
        HashMismatchError,
        SplitMismatchError,
    )

    try:
        variants = dict(
            parse_variant_checkpoint(value)
            for value in (args.variant_checkpoints or [])
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    config = EvaluateEffectModelConfig(
        checkpoint=Path(args.checkpoint),
        variant_checkpoints=variants,
        records_dir=Path(args.records_dir),
        cards_folders=resolve_cards_folders(args.cards_folders),
        variant_scripts=(
            Path(args.variant_scripts) if args.variant_scripts else None
        ),
        vocab_path=Path(args.vocab_path) if args.vocab_path else None,
        keyword_definitions=(
            Path(args.keyword_definitions) if args.keyword_definitions else None
        ),
        abilities_root=Path(args.abilities_root),
        sealed_encoder_checkpoint=Path(args.sealed_encoder_checkpoint),
    )
    try:
        report = evaluate(config)
    except (HashMismatchError, SplitMismatchError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    print(report.render())
    return 0 if report.ships else 1


# ── the table ───────────────────────────────────────────────────────────

_SUBCOMMAND_BUILDERS = (
    _build_vocab_parser,
    _extract_keyword_definitions_parser,
    _collect_coverage_parser,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
