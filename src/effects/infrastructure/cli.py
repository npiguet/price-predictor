"""CLI interface for the effects module (``python -m effects``).

Mirrors ``draft/infrastructure/cli.py``: ``build_parser`` adds one subparser per
command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and calls the
resolved ``func``. Each ``run_*`` imports its application module when it runs;
the parser itself imports only defaults, from modules that load no torch, so
``python -m effects --help`` never pays the torch import cost.
``tests/unit/effects/infrastructure/test_cli_parser_imports.py`` pins that.

Every flag and default here is contract, pinned by
``specs/023-ability-effect-model/contracts/cli.md`` and asserted by the
contract tests — a default that drifts changes what a corpus means.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from effects.application import collect_coverage, collect_variants
from effects.application.build_corpus import DEFAULT_GAME_DISJOINT_KEYWORDS
from effects.application.build_vocab import DEFAULT_TARGET_SIZE
from effects.application.train_effect_model import (
    DEFAULT_PREFETCH_WORKERS,
    DEFAULT_SHARDS_PER_EPOCH,
)
from effects.application.validate_corpus import Thresholds
from effects.domain.collection_caps import (
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_LEGALITY_RATE,
    DEFAULT_MANA_CAP,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_SHARE,
    CollectionCaps,
)
from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE
from effects.domain.variants import VARIANTS, Variant
from effects.infrastructure.default_paths import (
    DEFAULT_ABILITIES_ROOT,
    DEFAULT_CARDS_FOLDERS,
    DEFAULT_CHECKPOINT,
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FORGE_CARDS_PATH,
    DEFAULT_FORGE_TOKENSCRIPTS,
    DEFAULT_KEYWORD_DEFINITIONS,
    DEFAULT_MATCH_OUTCOMES,
    DEFAULT_RECORDS_DIR,
    DEFAULT_SCRIPT_VOCAB_PATH,
    DEFAULT_SEALED_ENCODER,
    DEFAULT_VARIANT_SCRIPTS,
    DEFAULT_VOCAB_PATH,
    DEFAULT_WIN_RATES,
)

logger = logging.getLogger(__name__)

_Config = TypeVar("_Config")


def config_from_args(cls: type[_Config], args: argparse.Namespace, **overrides: Any) -> _Config:
    """``cls`` built from the parsed flags whose dest names one of its fields.

    Every command's dests are its config's field names, so the config is read
    off the namespace rather than copied field by field, and a flag added to
    both reaches the run without a third edit. A field with no flag keeps the
    dataclass default. ``overrides`` supplies what a flag cannot give as
    parsed: a location resolved from several flags, a nested config, a dest
    spelled differently. An override naming no field is refused, so a typo
    cannot quietly fall back to the default.
    """
    names = {spec.name for spec in fields(cls) if spec.init}
    unknown = sorted(overrides.keys() - names)
    if unknown:
        raise TypeError(f"{cls.__name__} has no field {', '.join(unknown)}")
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return cls(**{**values, **overrides})


def _optional_path(text: str) -> Path | None:
    """A path flag's ``type`` where an empty value means unset, as omitting it does."""
    return Path(text) if text else None


# ── flags several commands share ────────────────────────────────────────


def _add_cards_folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-folder", action="append", dest="cards_folders", default=None,
        help=(
            "Converted source tree; repeatable. Defaults to "
            f"{' and '.join(DEFAULT_CARDS_FOLDERS)}."
        ),
    )


def resolve_cards_folders(values: list[str] | None) -> tuple[Path, ...]:
    """``--cards-folder`` values, or both converted trees."""
    chosen = values or list(DEFAULT_CARDS_FOLDERS)
    return tuple(Path(value) for value in chosen)


def _add_effect_records(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """``--effect-records``, defaulting to the raw shard root on every command."""
    parser.add_argument(
        "--effect-records", type=Path, default=DEFAULT_RECORDS_DIR, help=help_text,
    )


#: The name a depletion list takes inside the corpus it depleted.
HOLDOUT_LIST_NAME = "holdout-cards.txt"


def _add_holdout_sources(
    parser: argparse.ArgumentParser, *, excluded: str = "held-out cards",
) -> None:
    """The three ways a collector names the cards it keeps out of every deck.

    One definition for both collectors, so neither can accept a source the
    other refuses: a corpus depleted against one holdout and split against
    another is a silent corruption. ``excluded`` names what a checkpoint's
    holdout removes on this command.
    """
    parser.add_argument(
        "--split-from", type=_optional_path, default=None,
        help=(
            f"A checkpoint whose {excluded} are excluded from every deck. "
            "Without a holdout source a collection run contaminates the "
            "card-disjoint split of the model it feeds."
        ),
    )
    parser.add_argument(
        "--exclude-cards", type=_optional_path, default=None,
        help=(
            "File of card names, one per line, to keep out of every deck — "
            "from 'python -m effects holdout-cards'. The ordinary way to name "
            "the holdout: it needs no trained model, so this command no longer "
            "waits on a training run that follows it. Mutually exclusive with "
            "--split-from, which inherits an existing checkpoint's split "
            "instead."
        ),
    )
    parser.add_argument(
        "--training-corpus", type=Path, default=None,
        help=(
            f"Directory holding the training corpus AND the {HOLDOUT_LIST_NAME} "
            "it was depleted against. Shorthand for --effect-records DIR plus "
            f"--exclude-cards DIR/{HOLDOUT_LIST_NAME}, which belong together: a "
            "corpus depleted against one list and split against another is a "
            "silent corruption. Refused alongside either flag it implies."
        ),
    )


#: Java worker processes a collector runs, as both collectors' configs default.
DEFAULT_COLLECTOR_WORKERS = 12


def _add_collector_round_flags(
    parser: argparse.ArgumentParser, *, decks_per_round: int,
) -> None:
    """How much a collector plays per round, and on how many workers."""
    parser.add_argument(
        "--decks-per-round", type=int, default=decks_per_round,
        help=f"Decks played as matches per round (default: {decks_per_round})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_COLLECTOR_WORKERS,
        help=(
            "Parallel Java worker processes (default: "
            f"{DEFAULT_COLLECTOR_WORKERS})"
        ),
    )


def _add_holdout_flags(parser: argparse.ArgumentParser) -> None:
    """The holdout rule, on both commands that compute it.

    ``holdout-cards`` writes the depletion list and ``build-corpus`` splits the
    corpus; the two must be passed the same values, or pools are depleted
    against one holdout while the split uses another.
    """
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "An eligible text (a template, under --holdout-unit template) is "
            "held out when its crc32 %% 1000 is below this "
            f"(default: {HOLDOUT_PERMILLE})"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text (or template) more cards than this carry is never eligible "
            f"(default: {HOLDOUT_MAX_CARRIERS})"
        ),
    )
    _add_holdout_unit(parser, default="template")


def _add_holdout_unit(parser: argparse.ArgumentParser, *, default: str | None) -> None:
    """``--holdout-unit``, shared by every command that computes the holdout."""
    parser.add_argument(
        "--holdout-unit", choices=("template", "text"), default=default,
        help=(
            "What the holdout keys on: 'template' holds out every text sharing "
            "a masked template (digits, CARDNAME and descriptions masked) "
            "together; 'text' is feature 023's per-text rule"
            + (f" (default: {default})" if default else
               " (default: the corpus manifest's recorded unit)")
        ),
    )


def _add_variant(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    """``--variant``: the shipping model or one of its baselines."""
    parser.add_argument(
        "--variant", choices=VARIANTS, default=Variant.FULL.value, help=help_text,
    )


def _add_variant_scripts(
    parser: argparse.ArgumentParser, *,
    without: str = "no variant tree is read",
    path_type: Callable[[str], object] = _optional_path,
) -> None:
    """``--variant-scripts`` on a command that reads the stage-four tree.

    ``without`` says what an unset flag costs on this command.
    """
    parser.add_argument(
        "--variant-scripts", type=path_type, default=None,
        help=(
            "Stage four: the perturbed-script tree collect-variants wrote "
            f"(usually {DEFAULT_VARIANT_SCRIPTS}). Unset by default, and then "
            f"{without}"
        ),
    )


def _add_recorded_tokenizer_paths(parser: argparse.ArgumentParser) -> None:
    """``--vocab-path`` and ``--keyword-definitions`` on a checkpoint reader.

    Both default to None, read as the paths the checkpoint recorded from
    training (FR-097): a constant would tokenize with a vocabulary the model
    may never have seen.
    """
    for flag in ("--vocab-path", "--keyword-definitions"):
        parser.add_argument(
            flag, type=_optional_path, default=None,
            help="Defaults to the path the checkpoint recorded from training",
        )


def _add_cap_flags(parser: argparse.ArgumentParser) -> None:
    """The cap and budget flags every collecting supervisor carries."""
    parser.add_argument(
        "--mana-cap", type=int, default=DEFAULT_MANA_CAP,
        help=(
            "Records per unique mana ability, per game (default: "
            f"{DEFAULT_MANA_CAP}). A "
            "Mountain taps a dozen times a game for the same R, and the "
            "repeats observe a board that barely moved. Keyed on the mana "
            "produced too, so a dual land's two colours both record."
        ),
    )
    parser.add_argument(
        "--playability-rate", type=float, default=DEFAULT_PLAYABILITY_RATE,
        help=(
            "Fraction of decision-subkind logging points sampled (default: "
            f"{DEFAULT_PLAYABILITY_RATE:g}). The legality subkinds are not "
            "sampled — identical answers are coalesced instead."
        ),
    )
    parser.add_argument(
        "--interventions-per-game", type=int,
        default=DEFAULT_INTERVENTIONS_PER_GAME,
        help=(
            "Interventional resolutions per game, stage three (default: "
            f"{DEFAULT_INTERVENTIONS_PER_GAME})"
        ),
    )
    parser.add_argument(
        "--probes-per-game", type=int, default=DEFAULT_PROBES_PER_GAME,
        help=(
            "Damage-step probe forks per game (default: "
            f"{DEFAULT_PROBES_PER_GAME}). A budget, not a "
            "switch: it is spent only on keywords --probe-keywords names."
        ),
    )
    parser.add_argument(
        "--probe-keywords", type=str, default=DEFAULT_PROBE_KEYWORDS,
        help=(
            "Comma-separated damage-step keywords to probe (default: all "
            f"eight, {DEFAULT_PROBE_KEYWORDS}). Name a subset to narrow the "
            "probes; an empty string takes no fork whatever --probes-per-game "
            "says."
        ),
    )
    parser.add_argument(
        "--legality-rate", type=float, default=DEFAULT_LEGALITY_RATE,
        help=(
            "Fraction of the AI's what-if legality queries kept (default: "
            f"{DEFAULT_LEGALITY_RATE:g}, none). Real attack and block "
            "declarations are always written."
        ),
    )
    parser.add_argument(
        "--random-seat-share", type=float, default=DEFAULT_RANDOM_SEAT_SHARE,
        help=(
            "Share of matches in which one seat, chosen at random per match, "
            "is a random seat: with probability --random-seat-probability it "
            "plays, targets, attacks and blocks uniformly at random among the "
            "legal options instead of as the Forge AI, so it casts cards the "
            f"AI never would (default: {DEFAULT_RANDOM_SEAT_SHARE:g}, none)."
        ),
    )
    parser.add_argument(
        "--random-seat-probability", type=float, default=None,
        help=(
            "P, the chance the random seat draws at random at a decision "
            "point. Required when --random-seat-share is above 0; no default."
        ),
    )


def random_seat_error(args: argparse.Namespace) -> str | None:
    """Why the random-seat flags cannot run, or None when they can (FR-026).

    The same rules ``match-outcomes`` checks before any game: a share with no
    ``P`` is a seat whose behaviour nobody chose. Restated rather than imported,
    because ``effects`` never imports ``sealed.infrastructure``.
    """
    share = getattr(args, "random_seat_share", DEFAULT_RANDOM_SEAT_SHARE)
    probability = getattr(args, "random_seat_probability", None)
    if not 0.0 <= share <= 1.0:
        return f"--random-seat-share must be between 0 and 1, got {share}"
    if probability is not None and not 0.0 <= probability <= 1.0:
        return (
            f"--random-seat-probability must be between 0 and 1, got {probability}"
        )
    if share > 0 and probability is None:
        return "--random-seat-share above 0 needs --random-seat-probability"
    return None


def announce_probe_state(probe_keywords: str, probes_per_game: int) -> str:
    """The startup line every collecting supervisor prints about probes.

    ``--probes-per-game`` reads like the switch and is not: the budget is
    spent only on keywords ``--probe-keywords`` names. A run that empties that
    list collects a corpus with zero probe forks in it — the smoke corpus had
    exactly that, 55,296 records and no probe evidence at all — and nothing
    said so until gate 2 had nothing to check against. So the state is
    announced rather than left to the flag table.
    """
    named = [word.strip() for word in probe_keywords.split(",") if word.strip()]
    if not named:
        return (
            "Damage-step probes DISABLED: --probe-keywords is empty, so the "
            f"--probes-per-game {probes_per_game} budget buys no fork at all. "
            "Pass --probe-keywords <keyword>[,<keyword>...] to take any."
        )
    return (
        f"Damage-step probes: up to {probes_per_game} per game on "
        f"{', '.join(named)}"
    )


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
        "--vocab-path", type=_optional_path, default=None,
        help=(
            f"Output path (default: {DEFAULT_VOCAB_PATH}, or "
            f"{DEFAULT_SCRIPT_VOCAB_PATH} under --surface script)"
        ),
    )
    parser.add_argument(
        "--keyword-definitions", type=Path, default=DEFAULT_KEYWORD_DEFINITIONS,
        help=f"Keyword definitions to scan (default: {DEFAULT_KEYWORD_DEFINITIONS})",
    )
    parser.add_argument(
        "--target-size", type=int, default=DEFAULT_TARGET_SIZE,
        help=(
            "Post-truncate the corpus-frequency vocabulary (default: "
            f"{DEFAULT_TARGET_SIZE})"
        ),
    )


def run_build_vocab(args: argparse.Namespace) -> int:
    from effects.application.build_vocab import (
        BuildVocabConfig,
        EmptyCardsFolderError,
    )
    from effects.application.build_vocab import run as build_vocab

    config = config_from_args(
        BuildVocabConfig, args,
        cards_folders=resolve_cards_folders(args.cards_folders),
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
        "--output", type=Path, default=DEFAULT_KEYWORD_DEFINITIONS,
        help=f"Output path (default: {DEFAULT_KEYWORD_DEFINITIONS})",
    )


def run_extract_keyword_definitions(args: argparse.Namespace) -> int:
    from effects.application.extract_keyword_definitions import (
        ExtractKeywordDefinitionsConfig,
    )
    from effects.application.extract_keyword_definitions import (
        run as extract,
    )

    return extract(config_from_args(ExtractKeywordDefinitionsConfig, args))


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
    _add_effect_records(parser, help_text=(
        "Destination shard directory. Unlike on match-outcomes this has a "
        f"default ({DEFAULT_RECORDS_DIR}): collection is the whole point "
        "of this command."
    ))
    _add_cards_folder(parser)
    _add_holdout_sources(parser)
    parser.add_argument(
        "--only-cards", type=_optional_path, default=None,
        help=(
            "The held-out coverage round: the list 'holdout-cards' writes. "
            "Decks are built from these cards alone, plus basics, records are "
            "full strength, and the unit is the held-out ability text. "
            "Refused beside --exclude-cards, --training-corpus or --split-from"
        ),
    )
    parser.add_argument(
        "--min-text-games", type=int,
        default=collect_coverage.DEFAULT_MIN_TEXT_GAMES,
        help=(
            "With --only-cards: distinct games a held-out text must act in to "
            f"be satisfied (default: {collect_coverage.DEFAULT_MIN_TEXT_GAMES})"
        ),
    )
    parser.add_argument(
        "--target-records", type=int,
        default=collect_coverage.DEFAULT_TARGET_RECORDS,
        help=(
            "Per-card satisfaction goal (default: "
            f"{collect_coverage.DEFAULT_TARGET_RECORDS})"
        ),
    )
    parser.add_argument(
        "--no-progress-rounds", type=int,
        default=collect_coverage.DEFAULT_NO_PROGRESS_ROUNDS,
        help=(
            "Consecutive rounds without a new qualifying record before a card "
            f"retires (default: {collect_coverage.DEFAULT_NO_PROGRESS_ROUNDS}). "
            "This is what makes the run terminate."
        ),
    )
    _add_collector_round_flags(
        parser, decks_per_round=collect_coverage.DEFAULT_DECKS_PER_ROUND,
    )
    _add_cap_flags(parser)


def holdout_source(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """:func:`resolve_training_corpus`, refusing a second source beside it.

    Checked here, before any worker starts, so a collector refuses
    ``--split-from`` beside a depletion list as a message rather than as a
    traceback out of its first round.

    Raises:
        ValueError: As :func:`resolve_training_corpus`, or when ``--split-from``
            names a holdout the depletion list already names.
    """
    records, listed = resolve_training_corpus(args)
    if args.split_from and listed is not None:
        raise ValueError(collect_coverage.TWO_HOLDOUT_SOURCES)
    return records, listed


def resolve_training_corpus(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """``(records dir, depletion list)`` from ``--training-corpus`` or the pair.

    A corpus and the list that depleted it belong together — the list is what
    the corpus was built against, and naming them separately is how a run ends
    up depleted against one holdout and split against another. ``DIR`` holding
    both is the shape an operator naturally produces.

    Raises:
        ValueError: If ``--training-corpus`` is combined with either flag it
            implies (two sources for one answer), or if the directory holds no
            depletion list — running undepleted by accident is the failure this
            whole area already had once.
    """
    corpus = getattr(args, "training_corpus", None)
    if corpus is None:
        return args.effect_records, args.exclude_cards
    if args.effect_records != Path(DEFAULT_RECORDS_DIR) or args.exclude_cards:
        raise ValueError(
            "pass one source of the corpus location, not both: "
            "--training-corpus DIR means --effect-records DIR with its own "
            f"{HOLDOUT_LIST_NAME}"
        )
    listed = corpus / HOLDOUT_LIST_NAME
    if not listed.is_file():
        raise ValueError(
            f"{corpus} holds no {HOLDOUT_LIST_NAME}. A training corpus is "
            "depleted against a list, and without it this run would deck the "
            "held-out cards and contaminate the split. Write one with "
            "'python -m effects holdout-cards', or pass --effect-records and "
            "--exclude-cards separately if they genuinely live apart."
        )
    return corpus, listed


def coverage_config_from(args: argparse.Namespace):
    """The coverage config, with the corpus location already resolved."""
    from effects.application.collect_coverage import CollectCoverageConfig

    only_cards = getattr(args, "only_cards", None)
    if only_cards is not None:
        clashing = [
            flag for flag, value in (
                ("--exclude-cards", args.exclude_cards),
                ("--training-corpus", getattr(args, "training_corpus", None)),
                ("--split-from", args.split_from),
            ) if value
        ]
        if clashing:
            # FR-038: the round decks the held-out cards on purpose, and each of
            # these names a holdout to keep out of the decks — the opposite
            # instruction, so one of them is a mistake.
            raise ValueError(
                f"--only-cards decks the held-out cards; {', '.join(clashing)} "
                "keeps a holdout out of every deck. Pass one or the other."
            )
    records, listed = holdout_source(args)
    return config_from_args(
        CollectCoverageConfig, args,
        effect_records=records,
        exclude_cards=listed,
        cards_folders=resolve_cards_folders(args.cards_folders),
        caps=CollectionCaps.from_args(args),
    )


def run_collect_coverage(args: argparse.Namespace) -> int:
    from effects.application.collect_coverage import run as collect

    error = random_seat_error(args)
    if error is not None:
        print(f"collect-coverage: {error}")
        return 1
    try:
        # Before any worker starts: every refusal lives in the config.
        config = coverage_config_from(args)
    except ValueError as exc:
        print(f"collect-coverage: {exc}")
        return 1
    print(announce_probe_state(args.probe_keywords, args.probes_per_game))
    return collect(config)


# ── build-corpus ────────────────────────────────────────────────────────


def _class_mix(text: str) -> dict[str, float]:
    """``--class-mix`` as argparse wants it: parsed through the trainer's own
    ``parse_kind_mix``, so the two commands cannot disagree about the syntax
    or the class vocabulary. Raising ``ArgumentTypeError`` puts a bad value in
    the usage message rather than in a traceback.
    """
    from effects.domain.sampling import parse_kind_mix

    try:
        return parse_kind_mix(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _keyword_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


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
    _add_corpus_selection(parser)
    _add_game_disjoint_stratum(parser)
    parser.add_argument(
        "--shard-records", type=int, default=2000,
        help=(
            "Records per output shard after repacking, cut at game boundaries; "
            "0 keeps one output shard per source shard (default: 2000)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Selection seed (default: 42)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel shard readers (default: CPU count)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Report drift against an existing manifest and write nothing",
    )


def _add_corpus_inputs(parser: argparse.ArgumentParser) -> None:
    """What ``build-corpus`` reads, and where it writes."""
    parser.add_argument(
        "--records-dir", type=Path, default=DEFAULT_RECORDS_DIR,
        help=f"Raw corpus root; discovery recurses (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_CORPUS_OUTPUT,
        help=f"Curated dataset directory (default: {DEFAULT_CORPUS_OUTPUT})",
    )
    _add_cards_folder(parser)
    parser.add_argument(
        # A string for the same reason as --variant-scripts below.
        "--vocab-path", type=str, default=DEFAULT_VOCAB_PATH,
        help=(
            "Decides the encoding surface the rarity table's text keys are "
            f"built on (default: {DEFAULT_VOCAB_PATH})"
        ),
    )
    # Kept as typed: the manifest records the string, and a Path would
    # rewrite its separators on Windows.
    _add_variant_scripts(parser, path_type=str, without=(
        "a variant line resolves to no ability text, so it gets neither a "
        "rarity-table entry nor a per-text cap while every real ability "
        "gets both"
    ))
    parser.add_argument(
        "--keyword-definitions", type=_optional_path, default=DEFAULT_KEYWORD_DEFINITIONS,
        help=(
            "The keyword table the tokenizer expands with; the build fails when "
            "a display-name match would expand a non-keyword line "
            f"(default: {DEFAULT_KEYWORD_DEFINITIONS})"
        ),
    )
    parser.add_argument(
        "--forge-tokenscripts", type=_optional_path, default=DEFAULT_FORGE_TOKENSCRIPTS,
        help=(
            "Forge's raw token scripts, read to remap the old collector's "
            "token keys: a key naming a cardsfolder path no tree holds, whose "
            "stem is a token's printed name, is rewritten to the token script "
            f"it meant (default: {DEFAULT_FORGE_TOKENSCRIPTS})"
        ),
    )
    parser.add_argument(
        "--no-remap-token-keys", dest="remap_token_keys", action="store_false",
        help=(
            "Build without the token-key remap, keeping every key exactly as "
            "collected. Records of a token's abilities then resolve to no "
            "ability text at all, as they did before the remap existed"
        ),
    )


def _add_corpus_selection(parser: argparse.ArgumentParser) -> None:
    """Which records the training corpus and its validation sample keep."""
    parser.add_argument(
        "--text-cap", type=int, default=200,
        help=(
            "Max written training records per ability text per selection cell, "
            "repeats included; 0 means no cap (default: 200)"
        ),
    )
    parser.add_argument(
        "--reuse-cap", type=int, default=4,
        help=(
            "Max copies of one record toward a short rule family's share; 1 "
            "means no repeats (default: 4)"
        ),
    )
    parser.add_argument(
        "--class-mix", type=_class_mix, default=None,
        help=(
            "class=share,… on-disk class proportions (default: the training "
            "mixture)"
        ),
    )
    parser.add_argument(
        "--training-records", type=int, default=0,
        help=(
            "Total training records after capping, subsampled per class; 0 "
            "means no ceiling (default: 0)"
        ),
    )
    parser.add_argument(
        "--card-disjoint-text-cap", type=int, default=50,
        help=(
            "Max card-disjoint games admitted per held-out ability text; 0 "
            "means no cap (default: 50)"
        ),
    )
    parser.add_argument(
        "--max-events-per-record", type=int, default=64,
        help=(
            "A record carrying more events than this is refused as a game "
            "dump rather than an ability; 0 disables the event-flood rule "
            "(default: 64)"
        ),
    )
    parser.add_argument(
        "--validation-sample", type=int, default=2048,
        help=(
            "Records per stratum in validation/samples/, mixture-matched; the "
            "card-disjoint resolution classes draw from the gate-one slice; "
            "0 writes no sample (default: 2048)"
        ),
    )


def _add_game_disjoint_stratum(parser: argparse.ArgumentParser) -> None:
    """Which games the game-disjoint validation stratum takes (FR-044, FR-045)."""
    parser.add_argument(
        "--game-disjoint-share", type=float, default=0.01,
        help=(
            "A game naming no held-out card enters the game-disjoint stratum "
            "when crc32(game_id) mod 1e6 / 1e6 is below this, so a rebuild over "
            "a grown corpus keeps every game it placed (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--game-disjoint-keyword-share", type=float, default=0.15,
        help=(
            "The threshold instead for a game holding a combat gate 2 qualifies "
            "for one of --game-disjoint-keywords (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--game-disjoint-keywords", type=_keyword_list,
        default=DEFAULT_GAME_DISJOINT_KEYWORDS,
        help=(
            "Comma list of damage-step keywords raising a game's threshold "
            f"(default: {','.join(DEFAULT_GAME_DISJOINT_KEYWORDS)})"
        ),
    )


def run_build_corpus(args: argparse.Namespace) -> int:
    from effects.application import build_corpus as build_corpus_module
    from effects.application.build_corpus import BuildCorpusConfig, BuildCorpusError

    try:
        # Inside the try: the config validates its own counts on construction.
        config = config_from_args(
            BuildCorpusConfig, args,
            cards_folders=tuple(
                str(path) for path in resolve_cards_folders(args.cards_folders)
            ),
            max_events=args.max_events_per_record,
        )
        return build_corpus_module.build(config)
    except BuildCorpusError as exc:
        # An empty --records-dir, a holdout that selected nothing, a --class-mix
        # naming classes the corpus lacks: expected, reportable conditions, not
        # crashes. A bare ValueError is NOT caught -- `decide`'s cap-mismatch
        # guard and `CapHeap.merge`'s are internal invariants, and a traceback
        # is what those are worth.
        logger.error("%s", exc)
        return 1
    except FileNotFoundError as exc:
        # `--verify` against a directory with no manifest. Every sibling
        # condition gets a clean message; this one used to get a traceback.
        logger.error("%s", exc)
        return 1


# ── holdout-cards ───────────────────────────────────────────────────────


def _holdout_cards_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "holdout-cards",
        help=(
            "Write the depletion list: every card carrying a held-out ability "
            "text, for 'sealed generate-pools --exclude-cards'"
        ),
    )
    parser.set_defaults(func=run_holdout_cards)
    parser.add_argument(
        "--out", type=str, required=True,
        help="Destination file, one Forge canonical card name per line",
    )
    parser.add_argument(
        "--cards-folder", type=str, default=DEFAULT_CARDS_FOLDERS[0],
        help=f"Converted card tree (default: {DEFAULT_CARDS_FOLDERS[0]})",
    )
    parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json",
        help=(
            "MTGJSON dump, used only to write the list in Forge's printed "
            "spelling rather than the converted tree's lowercase "
            "(default: resources/AllPrintings.json)"
        ),
    )
    _add_holdout_flags(parser)


def run_holdout_cards(args: argparse.Namespace) -> int:
    """Write the list, and report what share of the corpus it removes.

    The share is the number an operator needs: it is what a depleted pool loses,
    and it is the knob the two holdout flags actually turn.
    """
    from effects.application.holdout_cards import (
        holdout_report,
        write_depletion_list,
    )
    from effects.application.train_effect_model import (
        load_card_files,
        load_card_texts,
        load_first_printings,
    )
    from effects.infrastructure.sidecar_io import SidecarCache, load_tree_texts

    cards_folder = Path(args.cards_folder)
    if not cards_folder.is_dir():
        print(f"no converted cards under {cards_folder}")
        return 1

    card_files = load_card_files(cards_folder)
    if not card_files:
        print(f"no converted cards under {cards_folder}")
        return 1

    sidecars = SidecarCache({"cardsfolder": cards_folder})
    texts_by_card = load_card_texts(card_files, sidecars)
    printings = Path(args.printings_path)
    canonical = load_first_printings(printings) if printings.exists() else {}
    if not canonical:
        print(
            f"No printings at {printings}: the list keeps the converted tree's "
            "lowercase spelling. Every comparison folds case, so depletion "
            "still works."
        )
    report = holdout_report(
        card_files, texts_by_card,
        permille=args.holdout_permille,
        max_carriers=args.holdout_max_carriers,
        unit=args.holdout_unit,
        canonical_names=canonical,
        # The converted token tree beside the card tree, as build-corpus reads it.
        token_texts=load_tree_texts(cards_folder.parent / "tokenscripts"),
    )
    written = write_depletion_list(report.names, Path(args.out))
    if report.unit == "template":
        print(f"held-out templates: {report.keys}")
    print(f"held-out texts: {report.texts}")
    print(
        f"held-out cards: {written} of {len(card_files)} converted "
        f"({100.0 * report.depleted_share:.1f}% depleted) -> {args.out}"
    )
    if not written:
        print(
            "Nothing to deplete. Raise --holdout-permille, or check that the "
            "converted tree has sidecars with script text."
        )
        return 1
    return 0


# ── field-coverage ──────────────────────────────────────────────────────


def _field_coverage_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "field-coverage",
        help=(
            "Report which record fields a collected corpus has ever actually "
            "carried, and which are always the same value"
        ),
    )
    parser.set_defaults(func=run_field_coverage)
    _add_effect_records(
        parser, help_text=f"Shard directory to read (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="List every field, not only the constant ones",
    )


def run_field_coverage(args: argparse.Namespace) -> int:
    """Name the fields a corpus never varied, and say which are expected.

    A field written as a fixed literal and one that a short run happened not to
    exercise look identical from a corpus, so the report separates them by the
    checked-in list rather than pretending to infer it. A constant field marked
    NEW is either newly broken or newly rare, and only the writer says which.
    """
    from effects.application.field_coverage import (
        KNOWN_CONSTANT_FIELDS,
        field_coverage,
    )
    from effects.infrastructure.record_io import iter_shards, read_shard

    root = args.effect_records
    shards = iter_shards(root)
    if not shards:
        print(f"no shards under {root}")
        return 1

    records = (record for shard in shards for record in read_shard(shard))
    coverage = field_coverage(records)
    if not coverage:
        print(f"no records under {root}")
        return 1

    rows = coverage.values() if args.all else [
        row for row in coverage.values() if row.constant
    ]
    print(f"{len(coverage)} fields over {root}\n")
    for row in rows:
        if not row.constant:
            print(f"  [varies] {row.path}   ({row.populated}/{row.seen})")
        elif row.path in KNOWN_CONSTANT_FIELDS:
            print(f"  [known ] {row.path}   (seen {row.seen})")
        else:
            print(f"  [NEW   ] {row.path}   (seen {row.seen})")

    stale = sorted(
        path for path in KNOWN_CONSTANT_FIELDS
        if path in coverage and not coverage[path].constant
    )
    if stale:
        print("\nlisted as constant but carrying data — remove from the list:")
        for path in stale:
            print(f"  {path}")
    return 0


# ── validate-corpus ─────────────────────────────────────────────────────


#: The help for each ``Thresholds`` field, in the order ``--help`` lists them.
#: The flag is the field's name, and its default and type are the field's own,
#: so a threshold cannot drift from the value the check reads; why each default
#: is what it is lives on the field. A field defaulting to None is a rate that
#: is measured and reported as [WATCH] unless a floor is passed: its healthy
#: value is not yet known, so a default floor would fail every run, which is
#: the failure mode the whole command exists to avoid.
_THRESHOLD_HELP: dict[str, str] = {
    "min_keyed_rate": (
        "Share of records naming a line that must resolve it to a printed "
        "key (default: {default})"
    ),
    "max_duplicate_rate": (
        "Share of a kind's records that may repeat an earlier record of "
        "the same kind in the same game (default: {default})"
    ),
    "max_unpaired_link_rate": (
        "Share of link ids that may lack one activation and one resolution "
        "half (default: {default})"
    ),
    "max_names_per_game": (
        "Distinct card names one game_id may show before it is reading as "
        "more than one game (default: {default})"
    ),
    "min_cost_evidence": (
        "Activation records needed before a cost channel reading zero "
        "means the collector rather than the pool (default: {default})"
    ),
    "turn_jump_tolerance": (
        "Turns a snapshot may sit below the highest already seen in its "
        "game before that reads as a game boundary (default: {default}). "
        "Deferred mana-reservoir flushes are exempt and the report says how "
        "many were exempted."
    ),
    "max_duplicate_event_rate": (
        "Share of events that may repeat another event in the same record "
        "(default: {default}). Distinct from --max-duplicate-rate, which "
        "compares whole records and read zero on a corpus that duplicated one "
        "event in seven"
    ),
    "max_trigger_fired_share": (
        "Share of trigger records that may report fired=true (default: "
        "{default}); the negative sampler targets ~1:1, and the report breaks "
        "the ratio down per evaluated mode"
    ),
    "min_zone_change_from_zone_rate": (
        "Share of zone_change events that must carry from_zone. Unset by "
        "default: the rate is measured and reported as [WATCH] without "
        "failing the run"
    ),
    "min_attributed_rate": (
        "Share of resolution events that must name a producing clause. "
        "Unset by default: measured and reported as [WATCH] without "
        "failing the run"
    ),
    "min_fork_attributed_rate": (
        "Share of events on fork records that must name a producing "
        "clause. Unset by default: measured and reported as [WATCH]. Kept "
        "apart from --min-attributed-rate because the fork collectors are "
        "different code, and the aggregate hid a fork path writing null "
        "on every event"
    ),
    "min_cause_rate": (
        "Share of the events whose schema row declares a cause that must "
        "populate it. Unset by default: measured and reported as [WATCH]. "
        "An empty cause is what makes two attackers' identical damage "
        "read as one outcome written twice"
    ),
    "min_rewrite_replaced_by_rate": (
        "Share of the rewrite records whose result ran an ability that "
        "must name it. Unset by default: measured and reported as "
        "[WATCH]. A replacement that substitutes an ability and cannot "
        "say which one is the shape that left this channel unreadable"
    ),
    "max_mirror_turn_disagreement_rate": (
        "Share of fork records whose turn may differ from the record they "
        "mirror (default: {default}). A fork is taken at the moment it "
        "mirrors, so a disagreement is a stale snapshot; it surfaced before "
        "as a backward turn jump in the game_id check, which blamed a game "
        "merge that had not happened"
    ),
}


def _add_threshold_flags(parser: argparse.ArgumentParser) -> None:
    """One flag per ``Thresholds`` field, defaulting to the field's own value."""
    defaults = {spec.name: spec.default for spec in fields(Thresholds)}
    for name, help_text in _THRESHOLD_HELP.items():
        default = defaults[name]
        parser.add_argument(
            "--" + name.replace("_", "-"),
            # A watched rate defaults to None and takes a share when set.
            type=float if default is None else type(default),
            default=default,
            help=help_text.format(default=default),
        )


def _validate_corpus_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-corpus",
        help=(
            "Check a shard directory against the corpus invariants and fail "
            "loudly on any that a collector is breaking"
        ),
    )
    parser.set_defaults(func=run_validate_corpus)
    _add_effect_records(
        parser, help_text=f"Shard directory to read (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help=(
            "Stop after this many records (default: 0, read everything). The "
            "point of this command is to run on the first few minutes of a "
            "pass, and every record id is held in memory while it runs."
        ),
    )
    _add_threshold_flags(parser)
    _add_cards_folder(parser)


def run_validate_corpus(args: argparse.Namespace) -> int:
    """Measure every corpus invariant over a window and exit non-zero on any breach.

    Meant for the first few minutes of a collection pass rather than for a
    finished corpus: the defects it looks for are all present in the first
    minute of shards, and finding them there costs minutes instead of the eight
    hours the first run spent producing 14.7M unusable records. Every invariant
    prints its measurement whether it passed or not, because the number is what
    tells an operator whether a fix worked or merely moved.
    """
    from effects.application.validate_corpus import (
        FieldPresence,
        read_window,
        validate_corpus,
    )
    from effects.infrastructure.record_io import iter_shards

    root = args.effect_records
    shards = iter_shards(root)
    if not shards:
        print(f"no shards under {root}")
        return 1

    thresholds = config_from_args(Thresholds, args)
    sidecars = _sidecars_for_validation(args)
    presence = FieldPresence()
    findings = validate_corpus(
        read_window(root, args.limit, presence), thresholds, sidecars, presence,
    )

    window = f" (first {args.limit} records)" if args.limit else ""
    print(f"{len(shards)} shards under {root}{window}\n")
    for finding in findings:
        for line in finding.lines():
            print(line)
    failed = [finding for finding in findings if not finding.ok]
    watched = sum(1 for finding in findings if finding.watched)
    judged = len(findings) - watched
    if failed:
        print(f"\n{len(failed)} of {judged} judged invariants broken:")
        for finding in failed:
            print(f"  {finding.name}")
        return 1
    print(f"\nall {judged} judged invariants hold ({watched} watched, not judged)")
    return 0


def _sidecars_for_validation(args: argparse.Namespace):
    """A ``SidecarCache`` over the converted trees, or None where none exist.

    None rather than an empty cache, because the two say different things in
    the report: a cache whose every lookup fails reads as a corpus that does
    not join, when the truth is that this machine has no converted tree to join
    against. The keyword-join check reports "not checked" for that, and holds
    only over the trees it could actually read.
    """
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    roots = sidecar_roots(
        path for path in resolve_cards_folders(args.cards_folders) if path.is_dir()
    )
    return SidecarCache(roots) if roots else None


# ── collect-variants ────────────────────────────────────────────────────


def _collect_variants_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "collect-variants",
        help=(
            "Perturb Forge card scripts by one parameter and play the results, "
            "so the corpus contains texts no real card prints"
        ),
    )
    parser.set_defaults(func=run_collect_variants)
    _add_effect_records(
        parser,
        help_text=f"Destination shard directory (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--forge-cards-path", type=Path, default=DEFAULT_FORGE_CARDS_PATH,
        help=(
            "Forge's *source* card scripts, not the converted ones: a variant "
            "is a Forge script Forge must be able to load"
        ),
    )
    parser.add_argument(
        "--variant-scripts", type=Path, default=DEFAULT_VARIANT_SCRIPTS,
        help=f"Where perturbed scripts are written (default: {DEFAULT_VARIANT_SCRIPTS})",
    )
    parser.add_argument(
        "--variant-volume", type=float,
        default=collect_variants.DEFAULT_VARIANT_VOLUME,
        help=(
            "Cap on how many variant *scripts* are generated, as a fraction of "
            "the real records already present (default: "
            f"{collect_variants.DEFAULT_VARIANT_VOLUME}). Expressed "
            "against the corpus so it scales with it; at any real corpus size "
            "it exceeds the number of perturbable scripts and is inert."
        ),
    )
    parser.add_argument(
        "--corpus-records", type=_optional_path, default=None,
        help=(
            "The real records --variant-volume is measured against (default: "
            "--effect-records). Point it at the corpus when variants are "
            "written to their own directory, or the cap is measured against an "
            "empty destination and allows nothing."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=collect_variants.RANDOM_SEED,
        help=(
            "Seeds both the perturbation and the deck build (default: "
            f"{collect_variants.RANDOM_SEED}). A variant round weights every "
            "variant equally and plays once, so a repeat run with the same seed "
            "regenerates the same scripts and decks and collects nothing; "
            "change it to top a corpus up."
        ),
    )
    _add_holdout_sources(parser, excluded="held-out cards, and their variants,")
    _add_collector_round_flags(
        parser, decks_per_round=collect_variants.DEFAULT_DECKS_PER_ROUND,
    )
    _add_cap_flags(parser)


def run_collect_variants(args: argparse.Namespace) -> int:
    from effects.application.collect_variants import CollectVariantsConfig
    from effects.application.collect_variants import run as collect

    error = random_seat_error(args)
    if error is not None:
        print(f"collect-variants: {error}")
        return 1
    try:
        records, listed = holdout_source(args)
    except ValueError as exc:
        print(f"collect-variants: {exc}")
        return 1
    print(announce_probe_state(args.probe_keywords, args.probes_per_game))
    return collect(config_from_args(
        CollectVariantsConfig, args,
        effect_records=records,
        exclude_cards=listed,
        caps=CollectionCaps.from_args(args),
    ))


# ── train-effect-model ──────────────────────────────────────────────────


def _train_effect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-effect-model",
        help="Train the ability encoder and effect head jointly from random init",
    )
    parser.set_defaults(func=run_train_effect_model)
    _add_training_inputs(parser)
    _add_model_shape(parser)
    _add_training_schedule(parser)


def _add_training_inputs(parser: argparse.ArgumentParser) -> None:
    """What the trainer reads, and where its checkpoints go."""
    parser.add_argument(
        "--corpus", type=str, required=True,
        help=(
            "A curated dataset directory built by `build-corpus`, and the "
            "trainer's only input: its manifest records the split, the "
            "holdout, the validation samples and the rarity table. Training "
            "straight from raw shards was withdrawn with FR-125."
        ),
    )
    _add_cards_folder(parser)
    _add_holdout_unit(parser, default=None)
    _add_variant_scripts(parser)
    parser.add_argument(
        "--split-from", type=_optional_path, default=None,
        help=(
            "Accepted for compatibility with older invocations of a --variant "
            "run; nothing is read from this path. The manifest of --corpus is "
            "authoritative for the split."
        ),
    )
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument(
        "--keyword-definitions", type=Path, default=DEFAULT_KEYWORD_DEFINITIONS,
    )
    parser.add_argument(
        "--model-output", type=_optional_path, default=None,
        help=(
            "Checkpoint directory (default: models/effects/effect-model/ for "
            "--variant full, models/effects/effect-model/{variant}/ otherwise)"
        ),
    )
    _add_variant(parser)


def _add_model_shape(parser: argparse.ArgumentParser) -> None:
    """The architecture and the losses it trains under; the checkpoint records them."""
    parser.add_argument("--e-dim", type=int, default=64)
    parser.add_argument(
        "--e-noise", type=float, default=0.1,
        help=(
            "Noise ratio on e, relative to the running covariance of e itself; "
            "ramps in over the first --steps-per-epoch steps (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--value-weight", type=float, default=0.05,
        help="Loss weight of the training-only value head (default: 0.05)",
    )
    parser.add_argument(
        "--encoder-layers", type=int, default=4,
        help="Ability-encoder layers, recorded in the checkpoint (default: 4)",
    )
    parser.add_argument(
        "--encoder-d-model", type=int, default=256,
        help=(
            "Ability-encoder width, recorded in the checkpoint; must divide by "
            "the encoder's head count (default: 256)"
        ),
    )
    parser.add_argument("--keyword-expand-p", type=float, default=0.25)
    parser.add_argument("--context-dropout", type=float, default=0.15)
    parser.add_argument(
        "--lethality-copies", type=int, default=0,
        help=(
            "Counterfactual copies per qualifying damage or -X/-X resolution in "
            "each training step, with the target's toughness and the stated "
            "amount redrawn and the death labelled by the rules (default: 0)"
        ),
    )
    parser.add_argument("--mlm-weight", type=float, default=0.1)
    parser.add_argument("--mlm-mask-prob", type=float, default=0.15)
    parser.add_argument("--api-weight", type=float, default=0.05)
    parser.add_argument(
        "--curriculum-epoch", type=int, default=3,
        help="Epoch whose first step enables the sparse field group (default: 3)",
    )


def _add_training_schedule(parser: argparse.ArgumentParser) -> None:
    """Batches, epochs, stopping and resuming."""
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--context-cache", action="store_true",
        help=(
            "Accepted and ignored (not implemented): the stop-gradient "
            "momentum cache is specified as the fallback for when live "
            "re-encoding exceeds the GPU budget, but the trainer's batcher "
            "has no cache and re-encodes context abilities live"
        ),
    )
    parser.add_argument("--cache-refresh", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=5000)
    parser.add_argument(
        "--shards-per-epoch", type=int, default=DEFAULT_SHARDS_PER_EPOCH,
        help=(
            "Record shards an epoch reads, one resident at a time, drawn at "
            "random across the whole corpus. --steps-per-epoch fixes how long "
            "an epoch takes; this fixes how many different shards those steps "
            f"are spread over (default: {DEFAULT_SHARDS_PER_EPOCH})"
        ),
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=DEFAULT_PREFETCH_WORKERS,
        help=(
            "Worker processes that read shards and prepare each step's host "
            "half (tokenizing, surfaces, targets) while this process trains; "
            "0 prepares them in the training process on a thread. Changes "
            "how fast a run goes, never what it computes "
            f"(default: {DEFAULT_PREFETCH_WORKERS})"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seeds weight init, batch planning and each epoch's shard draw. "
            "Drawn from the OS and reported at startup when omitted, so runs "
            "differ by default and any one of them can be repeated by passing "
            "back the seed it logged"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Epochs without a new card-disjoint best before stopping; 0 never stops early",
    )
    parser.add_argument(
        "--lr-scale", type=float, default=1.0,
        help="Factor on the learning rate; may differ from the run a --resume continues",
    )
    parser.add_argument(
        "--save-every-epoch", action="store_true",
        help="Also keep every epoch's checkpoint under <model-output>/epochs/",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Continue an interrupted run from the resume.pt it left in "
            "--model-output after its last completed epoch. Every setting that "
            "shapes the run must match the interrupted one, --seed included."
        ),
    )
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
        EmptyHoldoutError,
        SurfaceMismatchError,
    )
    from effects.domain.ability_encoder import ENCODER_N_HEADS

    # Refused here, before anything is read (FR-062): the encoder's attention
    # splits its width across a fixed head count.
    if args.encoder_d_model <= 0 or args.encoder_d_model % ENCODER_N_HEADS:
        logger.error(
            "--encoder-d-model %d is not divisible by the encoder's %d "
            "attention heads; pick a multiple of %d.",
            args.encoder_d_model, ENCODER_N_HEADS, ENCODER_N_HEADS,
        )
        return 2
    if args.encoder_layers <= 0:
        logger.error("--encoder-layers must be at least 1, got %d.", args.encoder_layers)
        return 2

    config = train_config_from(args)

    from effects.application.train_effect_model import run as train

    try:
        return train(config)
    except SurfaceMismatchError as exc:
        # The dataset's rarity table and --vocab-path disagree about the
        # encoding surface, which is only knowable once the manifest is read.
        logger.error("%s", exc)
        return 2
    except EmptyHoldoutError as exc:
        # Raised from inside the loop, once the corpus's card-disjoint sample
        # turns out to be empty. A corpus problem with an exit-code row of its
        # own, not a crash to hand an operator as a traceback.
        logger.error("%s", exc)
        return 2


def train_config_from(args: argparse.Namespace):
    """The trainer's config, assembled from parsed arguments."""
    from effects.application.train_effect_model import TrainEffectModelConfig

    return config_from_args(
        TrainEffectModelConfig, args,
        cards_folders=resolve_cards_folders(args.cards_folders),
    )


# ── encode-abilities ────────────────────────────────────────────────────


def _encode_abilities_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "encode-abilities",
        help="Compute the per-ability embedding cache from a trained checkpoint",
    )
    parser.set_defaults(func=run_encode_abilities)
    _add_variant(
        parser, help_text="Resolves the checkpoint and the output suffix together",
    )
    parser.add_argument(
        "--checkpoint", type=_optional_path, default=None,
        help="Overrides the checkpoint --variant resolves",
    )
    _add_cards_folder(parser)
    _add_variant_scripts(parser)
    _add_recorded_tokenizer_paths(parser)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove only the files this variant wrote, then re-encode",
    )


def run_encode_abilities(args: argparse.Namespace) -> int:
    from effects.application.encode_abilities import EncodeAbilitiesConfig
    from effects.application.encode_abilities import run as encode
    from effects.infrastructure.effect_model_store import HashMismatchError

    config = config_from_args(
        EncodeAbilitiesConfig, args,
        cards_folders=resolve_cards_folders(args.cards_folders),
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--variant-checkpoint", action="append", dest="variant_checkpoints",
        default=None, metavar="NAME=PATH",
        help="Baseline checkpoint to compare against; repeatable",
    )
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    _add_cards_folder(parser)
    _add_variant_scripts(parser, without=(
        "the tree the corpus's manifest records is read, if any"
    ))
    _add_recorded_tokenizer_paths(parser)
    parser.add_argument(
        "--corpus", type=_optional_path, default=None,
        help=(
            "A curated dataset directory (build-corpus); defaults to the one "
            "--checkpoint recorded, if any. Fails fast when it has been "
            "rebuilt since training."
        ),
    )
    parser.add_argument("--abilities-root", type=Path, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--win-rates", type=Path, default=DEFAULT_WIN_RATES,
        help="Per-card win-rate table the decodability battery reads",
    )


def run_evaluate_effect_model(args: argparse.Namespace) -> int:
    from effects.application.evaluate_effect_model import run as evaluate
    from effects.application.evaluation_context import (
        CorpusMismatchError,
        EvaluateEffectModelConfig,
        parse_variant_checkpoint,
    )
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

    config = config_from_args(
        EvaluateEffectModelConfig, args,
        variant_checkpoints=variants,
        cards_folders=resolve_cards_folders(args.cards_folders),
    )
    try:
        report = evaluate(config)
    except (
        HashMismatchError, SplitMismatchError, CorpusMismatchError,
        FileNotFoundError,
    ) as exc:
        logger.error("%s", exc)
        return 2
    print(report.render())
    # A failed gate is a result to record, not a reason to stop: the sweep runs
    # every arm's probes after evaluation whatever the verdicts.
    return 0


# ── scorer-smoke-test ───────────────────────────────────────────────────


def _scorer_smoke_test_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "scorer-smoke-test",
        help=(
            "Train the sealed scorer's Phase A on sealed vectors with pooled e "
            "spliced in, entirely under --scratch-dir"
        ),
    )
    parser.set_defaults(func=run_scorer_smoke_test)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Effect model whose cache's pooled e is written",
    )
    parser.add_argument(
        "--sealed-encoder-checkpoint", type=Path,
        default=DEFAULT_SEALED_ENCODER,
        help="Sealed encoder whose vectors pooled e is concatenated with",
    )
    parser.add_argument(
        "--abilities-root", type=Path, default=DEFAULT_ABILITIES_ROOT,
        help=(
            "Ability cache whose pooled e is spliced in: the one encode-abilities "
            f"wrote from --checkpoint (default: {DEFAULT_ABILITIES_ROOT})"
        ),
    )
    parser.add_argument(
        "--outcomes-path", type=Path, default=DEFAULT_MATCH_OUTCOMES,
        help=f"Match outcomes Phase A trains on (default: {DEFAULT_MATCH_OUTCOMES})",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, required=True,
        help="Where the vectors and the scorer checkpoint go; nothing is written elsewhere",
    )


def run_scorer_smoke_test(args: argparse.Namespace) -> int:
    from effects.application.scorer_smoke_test import ScorerSmokeTestConfig
    from effects.application.scorer_smoke_test import run as smoke_test

    config = config_from_args(ScorerSmokeTestConfig, args)
    try:
        result = smoke_test(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    print(
        f"scorer smoke test: {result.cards_written} cards written "
        f"({result.cards_without_e} without e); Phase A exited "
        f"{result.scorer_exit_code}; scorer checkpoints under {config.scorer_dir}"
    )
    return result.scorer_exit_code


# ── the table ───────────────────────────────────────────────────────────

_SUBCOMMAND_BUILDERS = (
    _build_vocab_parser,
    _extract_keyword_definitions_parser,
    _collect_coverage_parser,
    _build_corpus_parser,
    _field_coverage_parser,
    _holdout_cards_parser,
    _validate_corpus_parser,
    _collect_variants_parser,
    _train_effect_model_parser,
    _encode_abilities_parser,
    _evaluate_effect_model_parser,
    _scorer_smoke_test_parser,
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

    Reports carry card texts and arrows, and a redirected stdout on Windows
    defaults to cp1252, which cannot encode them. A stream that cannot be
    reconfigured, such as a test's ``StringIO``, is left as it is.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> None:
    write_output_as_utf8(sys.stdout, sys.stderr)
    # Every line is stamped, because every command here is one an operator
    # leaves running for hours: a progress line without a clock answers "is it
    # alive" and not "when did it last move", which is the question actually
    # being asked of a survey that is two hundred shards in.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
