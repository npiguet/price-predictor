"""CLI interface for the effects module (``python -m effects``).

Each ``run_*`` imports its application module when it runs, and the parser
imports nothing that loads torch, so ``--help`` never pays for it
(``tests/unit/effects/infrastructure/test_cli_parser_imports.py``). Every flag
and default is contract, pinned by ``specs/023-ability-effect-model/contracts/cli.md``:
a default that drifts changes what a corpus means, which is why a flag mirroring
a config field takes its default from the field (:func:`add_config_flags`).
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from effects.application.build_corpus import (
    DEFAULT_GAME_DISJOINT_KEYWORDS,
    BuildCorpusConfig,
    BuildCorpusError,
)
from effects.application.build_vocab import DEFAULT_TARGET_SIZE, EmptyCardsFolderError
from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.collect_variants import CollectVariantsConfig
from effects.application.collector_common import HOLDOUT_LIST_NAME, HoldoutSource
from effects.application.evaluation_context import CorpusMismatchError
from effects.application.holdout_cards import HoldoutCardsConfig
from effects.application.train_effect_model import (
    EmptyHoldoutError,
    HoldoutMismatchError,
    SurfaceMismatchError,
    TrainEffectModelConfig,
)
from effects.application.validate_corpus import Thresholds
from effects.domain.ability_tokenizer import Surface
from effects.domain.baselines import BASELINES, Baseline
from effects.domain.collection_caps import DEFAULT_RANDOM_SEAT_PROBABILITY, CollectionCaps
from effects.domain.default_paths import (
    DEFAULT_ABILITIES_ROOT,
    DEFAULT_CARDS_FOLDER_PATHS,
    DEFAULT_CARDS_FOLDERS,
    DEFAULT_CHECKPOINT,
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FORGE_CARDS_PATH,
    DEFAULT_KEYWORD_DEFINITIONS,
    DEFAULT_RECORDS_DIR,
    DEFAULT_SCRIPT_VOCAB_PATH,
    DEFAULT_VARIANT_SCRIPTS,
    DEFAULT_VOCAB_PATH,
    DEFAULT_WIN_RATES,
)
from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE, HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from price_predictor.domain.dataclass_defaults import field_default
from sealed.infrastructure.effect_collection_flags import (
    add_cap_flags,
    announce_probe_state,
    random_seat_error,
)

logger = logging.getLogger(__name__)

_Config = TypeVar("_Config")

#: Torch loads with this module, so its exceptions are named rather than imported.
_EFFECT_MODEL_STORE = "effects.infrastructure.effect_model_store"


def config_from_args(cls: type[_Config], args: argparse.Namespace, **overrides: Any) -> _Config:
    """``cls`` built from the parsed flags whose dest names one of its fields.

    A field with no flag keeps the dataclass default. ``overrides`` supplies
    what a flag cannot give as parsed; one naming no field is refused, so a
    typo cannot quietly fall back to the default.
    """
    names = {spec.name for spec in fields(cls) if spec.init}
    unknown = sorted(overrides.keys() - names)
    if unknown:
        raise TypeError(f"{cls.__name__} has no field {', '.join(unknown)}")
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return cls(**{**values, **overrides})


def add_config_flags(
    parser: argparse.ArgumentParser, config: type, *, only: Iterable[str] | None = None,
) -> None:
    """A ``--field-name`` flag per field of ``config``, with the field's own default.

    ``only`` picks the fields, in order; without it, every field no flag on
    ``parser`` claims yet. The type comes from the annotation and the help from
    ``metadata["help"]``; a bool field is a ``store_true`` switch, and any other
    non-scalar is refused, because its flag needs declaring by hand. Reads the
    class and never builds one: a config's ``__post_init__`` may load torch.
    """
    specs = {spec.name: spec for spec in fields(config)}
    hints = typing.get_type_hints(config)
    claimed = {action.dest for action in parser._actions}
    for name in only if only is not None else [n for n in specs if n not in claimed]:
        spec = specs[name]
        kind = _flag_type(config, name, hints[name])
        default = field_default(spec)
        if isinstance(default, Path):
            # Spelled as typed in --help; argparse parses a string default too.
            default = default.as_posix()
        flag, help_text = "--" + name.replace("_", "-"), spec.metadata.get("help")
        if kind is bool and default is False:
            parser.add_argument(flag, action="store_true", help=help_text)
        elif kind is not bool:
            parser.add_argument(flag, type=kind, default=default, help=help_text)
        else:
            raise TypeError(f"{config.__name__}.{name} defaults to True; declare its flag")


def _flag_type(config: type, name: str, hint: Any) -> Callable[[str], Any]:
    """The flag's parser: the scalar itself, or a path, where an optional one
    reads an empty value as None."""
    kinds = [kind for kind in typing.get_args(hint) or (hint,) if kind is not type(None)]
    if len(kinds) == 1 and kinds[0] in (int, float, str, bool):
        return kinds[0]
    if kinds == [Path]:
        return _optional_path if type(None) in typing.get_args(hint) else Path
    raise TypeError(f"{config.__name__}.{name} is {hint}; declare its flag")


def exits_with(codes: Mapping[type[Exception] | str, int]) -> Callable:
    """A handler whose expected failures log their message and exit with a code.

    The first entry the exception is an instance of decides the code; anything
    else propagates as a traceback, which is what an internal invariant is
    worth. A string key names a class as ``module.Qualname`` and matches the
    exception's MRO without importing it.
    """
    def decorate(handler: Callable[[argparse.Namespace], int]):
        @functools.wraps(handler)
        def run(args: argparse.Namespace) -> int:
            try:
                return handler(args)
            except Exception as exc:
                code = _exit_code(codes, exc)
                if code is None:
                    raise
                logger.error("%s", exc)
                return code

        run.exit_codes = codes
        return run

    return decorate


def _exit_code(codes: Mapping[type[Exception] | str, int], exc: Exception) -> int | None:
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in type(exc).__mro__}
    for kind, code in codes.items():
        if kind in names if isinstance(kind, str) else isinstance(exc, kind):
            return code
    return None


def _optional_path(text: str) -> Path | None:
    """A path flag's ``type`` where an empty value means unset, as omitting it does."""
    return Path(text) if text else None


# ── flags several commands share ────────────────────────────────────────


class _AppendCardsFolder(argparse.Action):
    """``--cards-folder``: repeatable, and the first use replaces the default.

    argparse's own ``append`` appends to the default, so both converted trees
    would always be read beside the ones named.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        chosen = getattr(namespace, self.dest)
        if chosen is self.default:
            chosen = ()
        setattr(namespace, self.dest, (*chosen, Path(values)))


def _add_cards_folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-folder", action=_AppendCardsFolder, dest="cards_folders",
        default=DEFAULT_CARDS_FOLDER_PATHS,
        help=f"Converted source tree; repeatable (default: {' and '.join(DEFAULT_CARDS_FOLDERS)})",
    )


def _add_effect_records(
    parser: argparse.ArgumentParser, *, help_text: str,
    default: str | None = DEFAULT_RECORDS_DIR,
) -> None:
    """``--effect-records``; a collector passes ``default=None`` so it can tell
    an explicit one beside ``--training-corpus`` from none."""
    parser.add_argument(
        "--effect-records", type=Path, default=default, help=help_text,
    )


def _add_holdout_sources(
    parser: argparse.ArgumentParser, *, excluded: str = "held-out cards",
) -> None:
    """The three ways a collector names the cards it keeps out of every deck.

    Shared by both collectors, so neither accepts a source the other refuses:
    a corpus depleted against one holdout and split against another is a
    silent corruption.
    """
    parser.add_argument(
        "--split-from", type=_optional_path, default=None,
        help=(
            f"A checkpoint whose {excluded} are excluded from every deck, so "
            "this run cannot contaminate the card-disjoint split it feeds"
        ),
    )
    parser.add_argument(
        "--exclude-cards", type=_optional_path, default=None,
        help=(
            "Card names to keep out of every deck, from 'holdout-cards'; needs "
            "no trained model. Mutually exclusive with --split-from"
        ),
    )
    parser.add_argument(
        "--training-corpus", type=Path, default=None,
        help=(
            f"A corpus directory and the {HOLDOUT_LIST_NAME} it was depleted "
            "against: --effect-records DIR plus --exclude-cards "
            f"DIR/{HOLDOUT_LIST_NAME} in one flag, so the two cannot disagree"
        ),
    )


def _add_holdout_flags(parser: argparse.ArgumentParser) -> None:
    """The holdout rule, on every command that computes it: ``holdout-cards``,
    ``build-corpus`` and ``collect-variants`` must be passed the same values."""
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "A text (or template) is held out when its crc32 %% 1000 is below "
            "this (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text (or template) more cards than this carry is never held out "
            "(default: %(default)s)"
        ),
    )
    _add_holdout_unit(parser, default=HoldoutRule().unit)


def _add_holdout_unit(
    parser: argparse.ArgumentParser, *, default: HoldoutUnit | None,
) -> None:
    parser.add_argument(
        "--holdout-unit", type=HoldoutUnit, choices=list(HoldoutUnit), default=default,
        help=(
            "'template' holds out every text sharing a masked template together; "
            "'text' is feature 023's per-text rule (default: "
            + ("%(default)s)" if default else "the corpus manifest's recorded unit)")
        ),
    )


def holdout_rule(args: argparse.Namespace) -> HoldoutRule:
    """The rule :func:`_add_holdout_flags` parsed."""
    return HoldoutRule(args.holdout_permille, args.holdout_max_carriers, args.holdout_unit)


def _add_variant(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--variant", choices=BASELINES, default=Baseline.FULL.value, help=help_text,
    )


def _add_variant_scripts(
    parser: argparse.ArgumentParser, *,
    without: str = "no variant tree is read",
    path_type: Callable[[str], object] = _optional_path,
) -> None:
    """``--variant-scripts``; ``without`` says what leaving it unset costs."""
    parser.add_argument(
        "--variant-scripts", type=path_type, default=None,
        help=(
            "The perturbed-script tree collect-variants wrote (usually "
            f"{DEFAULT_VARIANT_SCRIPTS}); unset, {without}"
        ),
    )


def _add_recorded_tokenizer_paths(parser: argparse.ArgumentParser) -> None:
    """Default to the paths the checkpoint recorded (FR-097): any other would
    tokenize with a vocabulary the model may never have seen."""
    for flag in ("--vocab-path", "--keyword-definitions"):
        parser.add_argument(
            flag, type=_optional_path, default=None,
            help="Defaults to the path the checkpoint recorded from training",
        )


# ── build-vocab ─────────────────────────────────────────────────────────


def _build_vocab_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "build-vocab",
        help="Build the effects tokenizer vocabulary from the converted corpus",
    )
    parser.set_defaults(func=run_build_vocab)
    parser.add_argument(
        "--surface", type=Surface, choices=list(Surface), default=Surface.PROSE,
        help=(
            "prose reads the converted text; script adds the sidecars' script "
            "lines and switches the --vocab-path default (default: %(default)s)"
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
        help="Keyword definitions to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--target-size", type=int, default=DEFAULT_TARGET_SIZE,
        help="Post-truncate the corpus-frequency vocabulary (default: %(default)s)",
    )


@exits_with({EmptyCardsFolderError: 1, ValueError: 2})
def run_build_vocab(args: argparse.Namespace) -> int:
    from effects.application.build_vocab import BuildVocabConfig
    from effects.application.build_vocab import run as build_vocab

    build_vocab(config_from_args(BuildVocabConfig, args))
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
        help="Output path (default: %(default)s)",
    )


def run_extract_keyword_definitions(args: argparse.Namespace) -> int:
    from effects.application.extract_keyword_definitions import (
        ExtractKeywordDefinitionsConfig,
    )
    from effects.application.extract_keyword_definitions import (
        run as extract,
    )

    return extract(config_from_args(ExtractKeywordDefinitionsConfig, args))


# ── the collectors ──────────────────────────────────────────────────────


def _collect_coverage_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "collect-coverage",
        help=(
            "Play weighted decks over the whole converted corpus until every "
            "card is satisfied or retired"
        ),
    )
    parser.set_defaults(func=run_collect_coverage)
    _add_effect_records(parser, default=None, help_text=(
        f"Destination shard directory (default: {DEFAULT_RECORDS_DIR}); unlike "
        "on match-outcomes, collection is the point of this command"
    ))
    _add_cards_folder(parser)
    _add_holdout_sources(parser)
    parser.add_argument(
        "--only-cards", type=_optional_path, default=None,
        help=(
            "The held-out coverage round over the list 'holdout-cards' writes: "
            "decks from these cards alone, counted per held-out text. Refused "
            "beside any other holdout source"
        ),
    )
    add_config_flags(parser, CollectCoverageConfig, only=(
        "min_text_games", "target_records", "no_progress_rounds",
        "decks_per_round", "workers",
    ))
    add_cap_flags(parser)


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
        parser, default=None,
        help_text=f"Destination shard directory (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--forge-cards-path", type=Path, default=DEFAULT_FORGE_CARDS_PATH,
        help="Forge's source card scripts: a variant is a script Forge must load",
    )
    parser.add_argument(
        "--variant-scripts", type=Path, default=DEFAULT_VARIANT_SCRIPTS,
        help="Where perturbed scripts are written (default: %(default)s)",
    )
    parser.add_argument(
        "--corpus-records", type=_optional_path, default=None,
        help=(
            "The real records --variant-volume is measured against (default: "
            "--effect-records); name the corpus when variants go to their own "
            "directory, or the cap allows nothing"
        ),
    )
    add_config_flags(parser, CollectVariantsConfig, only=("variant_volume", "seed"))
    _add_holdout_sources(parser, excluded="held-out cards, and their variants,")
    # A variant printing a held-out text is dropped under this rule, so it has
    # to be the one build-corpus is given.
    _add_cards_folder(parser)
    _add_holdout_flags(parser)
    add_config_flags(parser, CollectVariantsConfig, only=("decks_per_round", "workers"))
    add_cap_flags(parser)


def holdout_source(args: argparse.Namespace) -> HoldoutSource:
    """A collector's :class:`HoldoutSource`; raises ValueError on a
    contradictory one, before any worker starts."""
    return HoldoutSource.from_flags(
        effect_records=args.effect_records,
        exclude_cards=args.exclude_cards,
        split_from=args.split_from,
        training_corpus=args.training_corpus,
        only_cards=getattr(args, "only_cards", None),
    )


def collection_caps(args: argparse.Namespace) -> CollectionCaps:
    # An unset --random-seat-probability is None so a share above 0 without
    # one can be refused; with the share at 0 the worker never reads it.
    probability = args.random_seat_probability
    return config_from_args(
        CollectionCaps, args,
        random_seat_probability=(
            DEFAULT_RANDOM_SEAT_PROBABILITY if probability is None else probability
        ),
    )


def collector_config(
    config_type: type[_Config], args: argparse.Namespace, **overrides: Any,
) -> _Config:
    """A collector's config, with its corpus location and holdout resolved."""
    source = holdout_source(args)
    return config_from_args(
        config_type, args,
        effect_records=source.effect_records,
        exclude_cards=source.exclude_cards,
        caps=collection_caps(args),
        **overrides,
    )


def _run_collector(
    args: argparse.Namespace, command: str, config_type: type,
    collect: Callable[[Any], int], **overrides: Any,
) -> int:
    """Refuse a bad flag combination as a message before any worker starts."""
    error = random_seat_error(args)
    if error is None:
        try:
            config = collector_config(config_type, args, **overrides)
        except ValueError as exc:
            error = str(exc)
    if error is not None:
        print(f"{command}: {error}")
        return 1
    print(announce_probe_state(args.probe_keywords, args.probes_per_game))
    return collect(config)


def run_collect_coverage(args: argparse.Namespace) -> int:
    from effects.application.collect_coverage import run as collect

    return _run_collector(args, "collect-coverage", CollectCoverageConfig, collect)


def run_collect_variants(args: argparse.Namespace) -> int:
    from effects.application.collect_variants import run as collect

    return _run_collector(
        args, "collect-variants", CollectVariantsConfig, collect,
        holdout=holdout_rule(args),
    )


# ── build-corpus ────────────────────────────────────────────────────────


def _class_mix(text: str) -> dict[str, float]:
    """``--class-mix`` parsed through the trainer's own ``parse_kind_mix``, so
    the two cannot disagree; a bad value lands in the usage message."""
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
    parser.add_argument(
        "--class-mix", type=_class_mix, default=None,
        help="class=share,… on-disk class proportions (default: the training mixture)",
    )
    parser.add_argument(
        "--max-events-per-record", type=int, default=MAX_EVENTS_PER_RECORD,
        help=(
            "A record carrying more events than this is refused as a game dump "
            "rather than an ability; 0 disables the event-flood rule "
            "(default: %(default)s)"
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
    add_config_flags(parser, BuildCorpusConfig, only=(
        "text_cap", "reuse_cap", "training_records", "card_disjoint_text_cap",
        "validation_sample", "game_disjoint_share", "game_disjoint_keyword_share",
        "shard_records", "seed", "workers", "verify",
    ))


def _add_corpus_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--records-dir", type=Path, default=DEFAULT_RECORDS_DIR,
        help="Raw corpus root; discovery recurses (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_CORPUS_OUTPUT,
        help="Curated dataset directory (default: %(default)s)",
    )
    _add_cards_folder(parser)
    # Strings rather than paths: the manifest records them, and a Path would
    # rewrite their separators on Windows.
    parser.add_argument(
        "--vocab-path", type=str, default=DEFAULT_VOCAB_PATH,
        help=(
            "Decides the encoding surface the rarity table's text keys are "
            "built on (default: %(default)s)"
        ),
    )
    _add_variant_scripts(parser, path_type=str, without=(
        "a variant line resolves to no ability text, so it escapes the rarity "
        "table and the per-text cap"
    ))
    add_config_flags(parser, BuildCorpusConfig, only=("keyword_definitions",))


# A bare ValueError is NOT mapped: `decide`'s cap-mismatch guard and
# `CapHeap.merge`'s are internal invariants, worth a traceback.
@exits_with({BuildCorpusError: 1, FileNotFoundError: 1})
def run_build_corpus(args: argparse.Namespace) -> int:
    from effects.application import build_corpus as build_corpus_module

    return build_corpus_module.build(config_from_args(
        BuildCorpusConfig, args, max_events=args.max_events_per_record,
        holdout=holdout_rule(args),
    ))


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
        "--out", type=Path, required=True,
        help="Destination file, one Forge canonical card name per line",
    )
    _add_cards_folder(parser)
    add_config_flags(parser, HoldoutCardsConfig, only=("printings_path",))
    _add_holdout_flags(parser)


def run_holdout_cards(args: argparse.Namespace) -> int:
    """Write the list, and report the share of the corpus it removes: what a
    depleted pool loses, and the knob the two holdout flags turn."""
    from effects.application.holdout_cards import HoldoutCardsError
    from effects.application.holdout_cards import run as holdout_cards

    config = config_from_args(HoldoutCardsConfig, args, holdout=holdout_rule(args))
    try:
        report = holdout_cards(config)
    except HoldoutCardsError as exc:
        print(exc)
        return 1
    if report.unit is HoldoutUnit.TEMPLATE:
        print(f"held-out templates: {report.keys}")
    print(f"held-out texts: {report.texts}")
    print(
        f"held-out cards: {len(report.names)} of {report.converted_cards} converted "
        f"({100.0 * report.depleted_share:.1f}% depleted) -> {config.out}"
    )
    if not report.names:
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
    _add_effect_records(parser, help_text="Shard directory to read (default: %(default)s)")
    parser.add_argument(
        "--all", action="store_true",
        help="List every field, not only the constant ones",
    )


def run_field_coverage(args: argparse.Namespace) -> int:
    from effects.application.field_coverage import field_coverage, report_lines
    from effects.infrastructure.record_io import iter_shards, read_shard

    root = args.effect_records
    shards = iter_shards(root)
    if not shards:
        print(f"no shards under {root}")
        return 1
    coverage = field_coverage(record for shard in shards for record in read_shard(shard))
    if not coverage:
        print(f"no records under {root}")
        return 1
    print(f"{len(coverage)} fields over {root}\n")
    for line in report_lines(coverage, all_fields=args.all):
        print(line)
    return 0


# ── validate-corpus ─────────────────────────────────────────────────────


def _validate_corpus_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-corpus",
        help=(
            "Check a shard directory against the corpus invariants and fail "
            "loudly on any that a collector is breaking"
        ),
    )
    parser.set_defaults(func=run_validate_corpus)
    _add_effect_records(parser, help_text="Shard directory to read (default: %(default)s)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help=(
            "Stop after this many records, 0 reading everything; every record "
            "id is held in memory (default: %(default)s)"
        ),
    )
    add_config_flags(parser, Thresholds)
    _add_cards_folder(parser)


def run_validate_corpus(args: argparse.Namespace) -> int:
    """Every invariant prints its measurement whether it passed or not: the
    number is what tells an operator whether a fix worked or merely moved."""
    from effects.application.validate_corpus import FieldPresence, validate_corpus
    from effects.infrastructure.record_io import iter_shards, read_window

    root = args.effect_records
    shards = iter_shards(root)
    if not shards:
        print(f"no shards under {root}")
        return 1

    presence = FieldPresence()
    findings = validate_corpus(
        read_window(root, args.limit, presence.observe), config_from_args(Thresholds, args),
        _sidecars_for_validation(args), presence,
    )
    window = f" (first {args.limit} records)" if args.limit else ""
    print(f"{len(shards)} shards under {root}{window}\n")
    for finding in findings:
        print("\n".join(finding.lines()))
    print("\n" + "\n".join(findings.summary()))
    return 1 if findings.broken() else 0


def _sidecars_for_validation(args: argparse.Namespace):
    """A ``SidecarCache`` over the converted trees, or None where none exist.

    None rather than an empty cache: a cache whose every lookup fails reads as
    a corpus that does not join, when this machine merely has no tree to join
    against, and the keyword-join check reports None as "not checked".
    """
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    roots = sidecar_roots(path for path in args.cards_folders if path.is_dir())
    return SidecarCache(roots) if roots else None


# ── train-effect-model ──────────────────────────────────────────────────


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
    _add_holdout_unit(parser, default=None)
    _add_variant_scripts(parser)
    _add_variant(parser)
    add_config_flags(parser, TrainEffectModelConfig)


@exits_with({
    HoldoutMismatchError: 1,
    "effects.infrastructure.record_io.MixedGenerationsError": 1,
    SurfaceMismatchError: 2,
    EmptyHoldoutError: 2,
    f"{_EFFECT_MODEL_STORE}.ResumeMismatchError": 2,
    # A missing manifest or resume file.
    FileNotFoundError: 2,
})
def run_train_effect_model(args: argparse.Namespace) -> int:
    from effects.application.train_effect_model import run as train

    try:
        config = config_from_args(TrainEffectModelConfig, args)
    except ValueError as exc:
        # An encoder shape the config refuses; a ValueError out of training
        # itself is an invariant and keeps its traceback.
        logger.error("%s", exc)
        return 2
    return train(config)


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


@exits_with({f"{_EFFECT_MODEL_STORE}.HashMismatchError": 2, FileNotFoundError: 2})
def run_encode_abilities(args: argparse.Namespace) -> int:
    from effects.application.encode_abilities import EncodeAbilitiesConfig
    from effects.application.encode_abilities import run as encode

    encode(config_from_args(EncodeAbilitiesConfig, args))
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
        "--variant-checkpoint", action="append", dest="baseline_checkpoints",
        type=_baseline_checkpoint, default=None, metavar="NAME=PATH",
        help="Baseline checkpoint to compare against; repeatable",
    )
    _add_cards_folder(parser)
    _add_variant_scripts(parser, without=(
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
    parser.add_argument("--abilities-root", type=Path, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--win-rates", type=Path, default=DEFAULT_WIN_RATES,
        help="Per-card win-rate table the decodability battery reads",
    )


def _baseline_checkpoint(value: str):
    from effects.application.evaluation_context import parse_baseline_checkpoint

    try:
        return parse_baseline_checkpoint(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


@exits_with({
    f"{_EFFECT_MODEL_STORE}.HashMismatchError": 2,
    f"{_EFFECT_MODEL_STORE}.SplitMismatchError": 2,
    CorpusMismatchError: 2,
    FileNotFoundError: 2,
})
def run_evaluate_effect_model(args: argparse.Namespace) -> int:
    from effects.application.evaluate_effect_model import run as evaluate
    from effects.application.evaluation_context import EvaluateEffectModelConfig

    print(evaluate(config_from_args(
        EvaluateEffectModelConfig, args,
        baseline_checkpoints=dict(args.baseline_checkpoints or []),
    )).render())
    # A failed gate is a result to record, not a reason to stop: the sweep runs
    # every arm's probes after evaluation whatever the verdicts.
    return 0


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


def main() -> None:
    write_output_as_utf8(sys.stdout, sys.stderr)
    # Stamped, because these commands run for hours: a progress line without a
    # clock answers "is it alive" and not "when did it last move".
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
