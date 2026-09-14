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

from effects.domain.collection_caps import (
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_LEGALITY_RATE,
    DEFAULT_MANA_CAP,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_SNAPSHOT_TIERS,
    CollectionCaps,
    parse_snapshot_tiers,
)

logger = logging.getLogger(__name__)

# ── shared defaults (contracts/cli.md) ──────────────────────────────────
DEFAULT_RECORDS_DIR = "output/effects/records/"
DEFAULT_CARDS_FOLDERS = ("output/cardsfolder/", "output/tokenscripts/")
DEFAULT_KEYWORD_DEFINITIONS = "output/effects/keyword-definitions.json"
DEFAULT_VOCAB_PATH = "models/effects/vocab.txt"
DEFAULT_SCRIPT_VOCAB_PATH = "models/effects/vocab-script.txt"
DEFAULT_VARIANT_SCRIPTS = "output/effects/variant-scripts/"
DEFAULT_ABILITIES_ROOT = "output/effects/abilities/"
#: Restated rather than imported, so `--help` stays free of torch. The
#: trainer's own constants are the source; `test_holdout_cli.py` pins them.
HOLDOUT_PERMILLE = 20
HOLDOUT_MAX_CARRIERS = 8
MIN_HOLDOUT_RECORDS = 2000
DEFAULT_CHECKPOINT = "models/effects/effect-model/latest.pt"
DEFAULT_PRINTINGS = "resources/AllPrintings.json"

#: Thresholds `validate-corpus` holds a fresh shard directory to. Set from the
#: first collected corpus's measurements, so a run reproducing any of its
#: defects fails rather than squeaking through.
DEFAULT_MIN_KEYED_RATE = 0.95
DEFAULT_MAX_DUPLICATE_RATE = 0.02
DEFAULT_MAX_UNPAIRED_LINK_RATE = 0.02
DEFAULT_MAX_NAMES_PER_GAME = 120
DEFAULT_TURN_JUMP_TOLERANCE = 1
DEFAULT_MIN_COST_EVIDENCE = 200
DEFAULT_MAX_DUPLICATE_EVENT_RATE = 0.02
DEFAULT_MAX_TRIGGER_FIRED_SHARE = 0.65

#: The rates `validate-corpus` measures and does not judge unless asked.
#: Their healthy value is not yet known — a `zone_change` for a card *made*
#: rather than moved has no origin to report, and a hook that names no causing
#: object is not a collector that lost it — so a default floor would fail every
#: run, which is the failure mode the whole command exists to avoid.
DEFAULT_MIN_ZONE_CHANGE_FROM_ZONE_RATE = None
DEFAULT_MIN_ATTRIBUTED_RATE = None
DEFAULT_MIN_FORK_ATTRIBUTED_RATE = None
DEFAULT_MIN_CAUSE_RATE = None
DEFAULT_MIN_REWRITE_REPLACED_BY_RATE = None

#: Judged, and at zero: a fork is taken at the moment it mirrors, so the two
#: records describe one moment and there is no rate at which they may disagree
#: about which turn it is.
DEFAULT_MAX_MIRROR_TURN_DISAGREEMENT_RATE = 0.0

#: Caps and budgets, shared by every collecting supervisor (FR-028).
CAP_DEFAULTS: dict[str, object] = {
    "mana_cap": DEFAULT_MANA_CAP,
    "playability_rate": DEFAULT_PLAYABILITY_RATE,
    "interventions_per_game": DEFAULT_INTERVENTIONS_PER_GAME,
    "probes_per_game": DEFAULT_PROBES_PER_GAME,
    "probe_keywords": "",
    "legality_rate": DEFAULT_LEGALITY_RATE,
    "snapshot_tiers": DEFAULT_SNAPSHOT_TIERS,
}


def _add_cards_folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-folder", action="append", dest="cards_folders", default=None,
        help=(
            "Converted source tree; repeatable. Defaults to "
            f"{' and '.join(DEFAULT_CARDS_FOLDERS)}."
        ),
    )


def _snapshot_tiers(text: str) -> str:
    """`--snapshot-tiers` as argparse wants it: validated, still the string.

    The value travels to the JVM as a comma-separated property, so it stays a
    string; only its shape is checked here. Raising
    ``ArgumentTypeError`` puts the rule in the usage message rather than in a
    traceback, and catches the vector the JVM would otherwise accept silently
    and then throw on mid-game.
    """
    try:
        parse_snapshot_tiers(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def _add_cap_flags(parser: argparse.ArgumentParser) -> None:
    """The cap and budget flags every collecting supervisor carries."""
    parser.add_argument(
        "--mana-cap", type=int, default=CAP_DEFAULTS["mana_cap"],
        help=(
            "Records per unique mana ability, per game (default: 1). A "
            "Mountain taps a dozen times a game for the same R, and the "
            "repeats observe a board that barely moved. Keyed on the mana "
            "produced too, so a dual land's two colours both record."
        ),
    )
    parser.add_argument(
        "--playability-rate", type=float,
        default=CAP_DEFAULTS["playability_rate"],
        help=(
            "Fraction of decision-subkind logging points sampled (default: "
            "0.1). The legality subkinds are not sampled — identical answers "
            "are coalesced instead."
        ),
    )
    parser.add_argument(
        "--interventions-per-game", type=int,
        default=CAP_DEFAULTS["interventions_per_game"],
        help="Interventional resolutions per game, stage three (default: 2)",
    )
    parser.add_argument(
        "--probes-per-game", type=int, default=CAP_DEFAULTS["probes_per_game"],
        help=(
            "Damage-step probe forks per game (default: 2). A budget, not a "
            "switch: it buys nothing unless --probe-keywords names a keyword."
        ),
    )
    parser.add_argument(
        "--probe-keywords", type=str, default=CAP_DEFAULTS["probe_keywords"],
        help=(
            "Comma-separated canary-failing keywords to probe. REQUIRED to get "
            "any probe at all: empty (the default) takes no fork whatever "
            "--probes-per-game says, and the run reports zero probes hours "
            "later. e.g. --probe-keywords "
            "first_strike,double_strike,deathtouch,lifelink,trample,"
            "indestructible,wither,infect"
        ),
    )
    parser.add_argument(
        "--legality-rate", type=float, default=CAP_DEFAULTS["legality_rate"],
        help=(
            "Fraction of legality-subkind logging points kept, sampled after "
            "the dedup (default: 0.1). Separate from --playability-rate "
            "because the two subkinds arrive at very different volumes from "
            "the same priority pass."
        ),
    )
    parser.add_argument(
        "--snapshot-tiers", type=_snapshot_tiers,
        default=CAP_DEFAULTS["snapshot_tiers"],
        help=(
            "Snapshot inclusion depth, comma-separated (default: "
            f"{DEFAULT_SNAPSHOT_TIERS}). A prefix of 1,2,3,4: 1 referenced "
            "objects, 2 core (global, battlefield, command-zone effect "
            "cards), 3 the unreferenced stack, 4 unreferenced hands and "
            "graveyards. Run-level, so every collector renders at the same "
            "depth — a depth that varies by kind makes state.tiers a proxy "
            "for how the record was collected."
        ),
    )


def announce_probe_state(probe_keywords: str, probes_per_game: int) -> str:
    """The startup line every collecting supervisor prints about probes.

    ``--probes-per-game`` reads like the switch and is not: the budget is
    spent only on keywords ``--probe-keywords`` names, and it names none by
    default. A run launched without it collects a corpus with zero probe forks
    in it — the smoke corpus had exactly that, 55,296 records and no probe
    evidence at all — and nothing said so until gate 2 had nothing to check
    against. So the absence is announced rather than left to the flag table.
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
        "--exclude-cards", type=str, default=None,
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
        "--training-corpus", type=str, default=None,
        help=(
            "Directory holding the training corpus AND the holdout-cards.txt "
            "it was depleted against. Shorthand for --effect-records DIR plus "
            "--exclude-cards DIR/holdout-cards.txt, which belong together: a "
            "corpus depleted against one list and split against another is a "
            "silent corruption. Refused alongside either flag it implies."
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


#: The name a depletion list takes inside the corpus it depleted.
HOLDOUT_LIST_NAME = "holdout-cards.txt"


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
        return (
            Path(args.effect_records),
            Path(args.exclude_cards) if args.exclude_cards else None,
        )
    if args.effect_records != DEFAULT_RECORDS_DIR or args.exclude_cards:
        raise ValueError(
            "pass one source of the corpus location, not both: "
            "--training-corpus DIR means --effect-records DIR with its own "
            f"{HOLDOUT_LIST_NAME}"
        )
    corpus = Path(corpus)
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

    records, listed = resolve_training_corpus(args)
    return CollectCoverageConfig(
        effect_records=records,
        cards_folders=resolve_cards_folders(args.cards_folders),
        split_from=Path(args.split_from) if args.split_from else None,
        exclude_cards=listed,
        target_records=args.target_records,
        decks_per_round=args.decks_per_round,
        no_progress_rounds=args.no_progress_rounds,
        workers=args.workers,
        caps=CollectionCaps.from_args(args),
    )


def run_collect_coverage(args: argparse.Namespace) -> int:
    from effects.application.collect_coverage import run as collect

    print(announce_probe_state(args.probe_keywords, args.probes_per_game))
    return collect(coverage_config_from(args))


# ── build-corpus ────────────────────────────────────────────────────────

#: Kept apart from DEFAULT_RECORDS_DIR: this is the *curated dataset*
#: directory build-corpus writes to, not the raw shard root it reads from
#: (specs/2026-09-05-ability-effect-model.md § Curated corpus).
DEFAULT_CORPUS_OUTPUT = "output/effects/corpus/"


def _class_mix(text: str) -> dict[str, float]:
    """``--class-mix`` as argparse wants it: parsed through the trainer's own
    ``parse_kind_mix``, so the two commands cannot disagree about the syntax
    or the class vocabulary. Raising ``ArgumentTypeError`` puts a bad value in
    the usage message rather than in a traceback (the same trade
    ``_snapshot_tiers`` above makes).
    """
    from effects.application.train_effect_model import parse_kind_mix

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
    parser.add_argument(
        "--records-dir", type=str, default=DEFAULT_RECORDS_DIR,
        help=f"Raw corpus root; discovery recurses (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_CORPUS_OUTPUT,
        help=f"Curated dataset directory (default: {DEFAULT_CORPUS_OUTPUT})",
    )
    _add_cards_folder(parser)
    parser.add_argument(
        "--vocab-path", type=str, default=DEFAULT_VOCAB_PATH,
        help=(
            "Decides the encoding surface the rarity table's text keys are "
            f"built on (default: {DEFAULT_VOCAB_PATH})"
        ),
    )
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "Holdout share of eligible ability texts "
            f"(default: {HOLDOUT_PERMILLE})"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text more cards than this carry stays out of the holdout "
            f"(default: {HOLDOUT_MAX_CARRIERS})"
        ),
    )
    parser.add_argument(
        "--text-cap", type=int, default=200,
        help="Max training records kept per unique ability text (default: 200)",
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
        "--game-disjoint-games", type=int, default=1000,
        help="Games withheld for the game-disjoint stratum (default: 1000)",
    )
    parser.add_argument(
        "--card-disjoint-text-cap", type=int, default=50,
        help=(
            "Max card-disjoint games admitted per held-out ability text "
            "(default: 50)"
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


def run_build_corpus(args: argparse.Namespace) -> int:
    from effects.application.build_corpus import BuildCorpusConfig
    from effects.application.build_corpus import build as build_corpus

    config = BuildCorpusConfig(
        records_dir=Path(args.records_dir),
        output=Path(args.output),
        cards_folders=tuple(
            str(path) for path in resolve_cards_folders(args.cards_folders)
        ),
        vocab_path=args.vocab_path,
        holdout_permille=args.holdout_permille,
        holdout_max_carriers=args.holdout_max_carriers,
        text_cap=args.text_cap,
        class_mix=args.class_mix,
        training_records=args.training_records,
        game_disjoint_target=args.game_disjoint_games,
        card_disjoint_text_cap=args.card_disjoint_text_cap,
        seed=args.seed,
        workers=args.workers,
        verify=args.verify,
    )
    try:
        return build_corpus(config)
    except ValueError as exc:
        # e.g. an empty --records-dir (run_survey's "no shards to build a
        # corpus from"): an expected, reportable condition, not a crash.
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
        "--cards-folder", type=str, default="output/cardsfolder/",
        help="Converted card tree (default: output/cardsfolder/)",
    )
    parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json",
        help=(
            "MTGJSON dump, used only to write the list in Forge's printed "
            "spelling rather than the converted tree's lowercase "
            "(default: resources/AllPrintings.json)"
        ),
    )
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "An eligible text is held out when crc32(text) %% 1000 is below "
            f"this (default: {HOLDOUT_PERMILLE})"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text more cards than this carry is never eligible "
            f"(default: {HOLDOUT_MAX_CARRIERS})"
        ),
    )


def run_holdout_cards(args: argparse.Namespace) -> int:
    """Write the list, and report what share of the corpus it removes.

    The share is the number an operator needs: it is what a depleted pool loses,
    and it is the knob the two holdout flags actually turn.
    """
    from effects.application.holdout_cards import (
        depletion_list,
        write_depletion_list,
    )
    from effects.application.train_effect_model import (
        load_card_files,
        load_card_texts,
        load_first_printings,
    )
    from effects.infrastructure.sidecar_io import SidecarCache

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
    names = depletion_list(
        card_files, texts_by_card,
        permille=args.holdout_permille,
        max_carriers=args.holdout_max_carriers,
        canonical_names=canonical,
    )
    written = write_depletion_list(names, Path(args.out))
    share = 100.0 * written / len(card_files)
    print(
        f"{written} of {len(card_files)} cards ({share:.1f}%) carry a held-out "
        f"text -> {args.out}"
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
    parser.add_argument(
        "--effect-records", type=str, default=DEFAULT_RECORDS_DIR,
        help=f"Shard directory to read (default: {DEFAULT_RECORDS_DIR})",
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

    root = Path(args.effect_records)
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


def _validate_corpus_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-corpus",
        help=(
            "Check a shard directory against the corpus invariants and fail "
            "loudly on any that a collector is breaking"
        ),
    )
    parser.set_defaults(func=run_validate_corpus)
    parser.add_argument(
        "--effect-records", type=str, default=DEFAULT_RECORDS_DIR,
        help=f"Shard directory to read (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help=(
            "Stop after this many records (default: 0, read everything). The "
            "point of this command is to run on the first few minutes of a "
            "pass, and every record id is held in memory while it runs."
        ),
    )
    parser.add_argument(
        "--min-keyed-rate", type=float, default=DEFAULT_MIN_KEYED_RATE,
        help=(
            "Share of records naming a line that must resolve it to a printed "
            f"key (default: {DEFAULT_MIN_KEYED_RATE})"
        ),
    )
    parser.add_argument(
        "--max-duplicate-rate", type=float, default=DEFAULT_MAX_DUPLICATE_RATE,
        help=(
            "Share of a kind's records that may repeat an earlier record of "
            f"the same kind in the same game (default: {DEFAULT_MAX_DUPLICATE_RATE})"
        ),
    )
    parser.add_argument(
        "--max-unpaired-link-rate", type=float,
        default=DEFAULT_MAX_UNPAIRED_LINK_RATE,
        help=(
            "Share of link ids that may lack one activation and one resolution "
            f"half (default: {DEFAULT_MAX_UNPAIRED_LINK_RATE})"
        ),
    )
    parser.add_argument(
        "--max-names-per-game", type=int, default=DEFAULT_MAX_NAMES_PER_GAME,
        help=(
            "Distinct card names one game_id may show before it is reading as "
            f"more than one game (default: {DEFAULT_MAX_NAMES_PER_GAME})"
        ),
    )
    parser.add_argument(
        "--min-cost-evidence", type=int, default=DEFAULT_MIN_COST_EVIDENCE,
        help=(
            "Activation records needed before a cost channel reading zero "
            "means the collector rather than the pool (default: "
            f"{DEFAULT_MIN_COST_EVIDENCE})"
        ),
    )
    parser.add_argument(
        "--turn-jump-tolerance", type=int, default=DEFAULT_TURN_JUMP_TOLERANCE,
        help=(
            "Turns a snapshot may sit below the highest already seen in its "
            f"game before that reads as a game boundary (default: "
            f"{DEFAULT_TURN_JUMP_TOLERANCE}). Deferred mana-reservoir flushes "
            "are exempt and the report says how many were exempted."
        ),
    )
    parser.add_argument(
        "--max-duplicate-event-rate", type=float,
        default=DEFAULT_MAX_DUPLICATE_EVENT_RATE,
        help=(
            "Share of events that may repeat another event in the same record "
            f"(default: {DEFAULT_MAX_DUPLICATE_EVENT_RATE}). Distinct from "
            "--max-duplicate-rate, which compares whole records and read zero "
            "on a corpus that duplicated one event in seven"
        ),
    )
    parser.add_argument(
        "--max-trigger-fired-share", type=float,
        default=DEFAULT_MAX_TRIGGER_FIRED_SHARE,
        help=(
            "Share of trigger records that may report fired=true (default: "
            f"{DEFAULT_MAX_TRIGGER_FIRED_SHARE}); the negative sampler targets "
            "~1:1, and the report breaks the ratio down per evaluated mode"
        ),
    )
    parser.add_argument(
        "--min-zone-change-from-zone-rate", type=float,
        default=DEFAULT_MIN_ZONE_CHANGE_FROM_ZONE_RATE,
        help=(
            "Share of zone_change events that must carry from_zone. Unset by "
            "default: the rate is measured and reported as [WATCH] without "
            "failing the run"
        ),
    )
    parser.add_argument(
        "--min-attributed-rate", type=float,
        default=DEFAULT_MIN_ATTRIBUTED_RATE,
        help=(
            "Share of resolution events that must name a producing clause. "
            "Unset by default: measured and reported as [WATCH] without "
            "failing the run"
        ),
    )
    parser.add_argument(
        "--min-fork-attributed-rate", type=float,
        default=DEFAULT_MIN_FORK_ATTRIBUTED_RATE,
        help=(
            "Share of events on fork records that must name a producing "
            "clause. Unset by default: measured and reported as [WATCH]. Kept "
            "apart from --min-attributed-rate because the fork collectors are "
            "different code, and the aggregate hid a fork path writing null "
            "on every event"
        ),
    )
    parser.add_argument(
        "--min-cause-rate", type=float, default=DEFAULT_MIN_CAUSE_RATE,
        help=(
            "Share of the events whose schema row declares a cause that must "
            "populate it. Unset by default: measured and reported as [WATCH]. "
            "An empty cause is what makes two attackers' identical damage "
            "read as one outcome written twice"
        ),
    )
    parser.add_argument(
        "--min-rewrite-replaced-by-rate", type=float,
        default=DEFAULT_MIN_REWRITE_REPLACED_BY_RATE,
        help=(
            "Share of the rewrite records whose result ran an ability that "
            "must name it. Unset by default: measured and reported as "
            "[WATCH]. A replacement that substitutes an ability and cannot "
            "say which one is the shape that left this channel unreadable"
        ),
    )
    parser.add_argument(
        "--max-mirror-turn-disagreement-rate", type=float,
        default=DEFAULT_MAX_MIRROR_TURN_DISAGREEMENT_RATE,
        help=(
            "Share of fork records whose turn may differ from the record they "
            f"mirror (default: {DEFAULT_MAX_MIRROR_TURN_DISAGREEMENT_RATE}). "
            "A fork is taken at the moment it mirrors, so a disagreement is a "
            "stale snapshot; it surfaced before as a backward turn jump in "
            "the game_id check, which blamed a game merge that had not "
            "happened"
        ),
    )
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
        Thresholds,
        read_window,
        validate_corpus,
    )
    from effects.infrastructure.record_io import iter_shards

    root = Path(args.effect_records)
    shards = iter_shards(root)
    if not shards:
        print(f"no shards under {root}")
        return 1

    thresholds = Thresholds(
        min_keyed_rate=args.min_keyed_rate,
        max_duplicate_rate=args.max_duplicate_rate,
        max_unpaired_link_rate=args.max_unpaired_link_rate,
        max_names_per_game=args.max_names_per_game,
        min_cost_evidence=args.min_cost_evidence,
        turn_jump_tolerance=args.turn_jump_tolerance,
        max_duplicate_event_rate=args.max_duplicate_event_rate,
        max_trigger_fired_share=args.max_trigger_fired_share,
        min_zone_change_from_zone_rate=args.min_zone_change_from_zone_rate,
        min_attributed_rate=args.min_attributed_rate,
        min_fork_attributed_rate=args.min_fork_attributed_rate,
        min_cause_rate=args.min_cause_rate,
        min_rewrite_replaced_by_rate=args.min_rewrite_replaced_by_rate,
        max_mirror_turn_disagreement_rate=args.max_mirror_turn_disagreement_rate,
    )
    sidecars = _sidecars_for_validation(args)
    findings = validate_corpus(read_window(root, args.limit), thresholds, sidecars)

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
    from effects.infrastructure.sidecar_io import SidecarCache

    roots = {
        path.name: path
        for path in resolve_cards_folders(args.cards_folders)
        if path.is_dir()
    }
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
    parser.add_argument(
        "--effect-records", type=str, default=DEFAULT_RECORDS_DIR,
        help=f"Destination shard directory (default: {DEFAULT_RECORDS_DIR})",
    )
    parser.add_argument(
        "--forge-cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
        help=(
            "Forge's *source* card scripts, not the converted ones: a variant "
            "is a Forge script Forge must be able to load"
        ),
    )
    parser.add_argument(
        "--variant-scripts", type=str, default=DEFAULT_VARIANT_SCRIPTS,
        help=f"Where perturbed scripts are written (default: {DEFAULT_VARIANT_SCRIPTS})",
    )
    parser.add_argument(
        "--variant-volume", type=float, default=0.2,
        help=(
            "Cap on variant records as a fraction of the real records already "
            "present (default: 0.2). Expressed against the corpus so it scales "
            "with it."
        ),
    )
    parser.add_argument("--decks-per-round", type=int, default=500)
    parser.add_argument(
        "--split-from", type=str, default=None,
        help=(
            "A checkpoint whose held-out cards, and their variants, are "
            "excluded"
        ),
    )
    parser.add_argument(
        "--exclude-cards", type=str, default=None,
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
        "--training-corpus", type=str, default=None,
        help=(
            "Directory holding the training corpus AND the holdout-cards.txt "
            "it was depleted against. Shorthand for --effect-records DIR plus "
            "--exclude-cards DIR/holdout-cards.txt, which belong together: a "
            "corpus depleted against one list and split against another is a "
            "silent corruption. Refused alongside either flag it implies."
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    _add_cap_flags(parser)


def run_collect_variants(args: argparse.Namespace) -> int:
    from effects.application.collect_variants import CollectVariantsConfig
    from effects.application.collect_variants import run as collect

    print(announce_probe_state(args.probe_keywords, args.probes_per_game))
    records, listed = resolve_training_corpus(args)
    return collect(CollectVariantsConfig(
        effect_records=records,
        forge_cards_path=Path(args.forge_cards_path),
        variant_scripts=Path(args.variant_scripts),
        variant_volume=args.variant_volume,
        decks_per_round=args.decks_per_round,
        split_from=Path(args.split_from) if args.split_from else None,
        exclude_cards=listed,
        workers=args.workers,
        caps=CollectionCaps.from_args(args),
    ))


# ── train-effect-model ──────────────────────────────────────────────────


def _train_effect_model_parser(subparsers) -> None:
    from effects.application.train_effect_model import (
        DEFAULT_SHARDS_PER_EPOCH,
        RESERVED_VALIDATION_SHARDS,
    )

    parser = subparsers.add_parser(
        "train-effect-model",
        help="Train the ability encoder and effect head jointly from random init",
    )
    parser.set_defaults(func=run_train_effect_model)
    parser.add_argument("--records-dir", type=str, default=DEFAULT_RECORDS_DIR)
    parser.add_argument(
        "--corpus", type=str, default=None,
        help=(
            "A curated dataset directory built by `build-corpus`. Its "
            "manifest already records the split, holdout and rarity table, "
            "so --records-dir, --reserved-shards, --split-from, "
            "--holdout-permille and --holdout-max-carriers cannot be passed "
            "alongside it."
        ),
    )
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
    parser.add_argument(
        "--shards-per-epoch", type=int, default=DEFAULT_SHARDS_PER_EPOCH,
        help=(
            "Record shards an epoch reads, one resident at a time. The walk "
            "advances each epoch and wraps, so a long run covers the corpus "
            f"rather than its opening slice (default: {DEFAULT_SHARDS_PER_EPOCH})"
        ),
    )
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "An eligible ability text is held out when crc32(text) %% 1000 is "
            "below this. Must match the value the corpus was depleted against "
            f"(default: {HOLDOUT_PERMILLE})"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text more cards than this carry is never eligible for the "
            "holdout, so depleting it cannot empty the corpus "
            f"(default: {HOLDOUT_MAX_CARRIERS})"
        ),
    )
    parser.add_argument(
        "--min-holdout-records", type=int, default=MIN_HOLDOUT_RECORDS,
        help=(
            "Warn when the card-disjoint stratum holds fewer unique-text "
            "resolution records than this; empty is a hard failure "
            f"(default: {MIN_HOLDOUT_RECORDS})"
        ),
    )
    parser.add_argument(
        "--reserved-shards", type=int, default=RESERVED_VALIDATION_SHARDS,
        help=(
            "Shards held back for validation and never trained on; their games "
            "supply both validation strata "
            f"(default: {RESERVED_VALIDATION_SHARDS})"
        ),
    )
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
        require_split_from,
    )

    split_from = Path(args.split_from) if args.split_from else None
    try:
        require_split_from(args.variant, split_from, corpus=args.corpus)
    except MissingSplitError as exc:
        logger.error("%s", exc)
        return 2

    config = train_config_from(args)
    from effects.application.train_effect_model import run as train

    return train(config)


def train_config_from(args: argparse.Namespace):
    """The trainer's config, assembled from parsed arguments.

    Imported here rather than at module scope so ``--help`` stays free of torch.
    """
    from effects.application.train_effect_model import TrainEffectModelConfig

    return TrainEffectModelConfig(
        records_dir=Path(args.records_dir),
        corpus=args.corpus,
        cards_folders=resolve_cards_folders(args.cards_folders),
        variant_scripts=(
            Path(args.variant_scripts) if args.variant_scripts else None
        ),
        split_from=Path(args.split_from) if args.split_from else None,
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
        shards_per_epoch=args.shards_per_epoch,
        reserved_shards=args.reserved_shards,
        epochs=args.epochs,
        patience=args.patience,
        withhold_keyword=args.withhold_keyword,
        holdout_permille=args.holdout_permille,
        holdout_max_carriers=args.holdout_max_carriers,
        min_holdout_records=args.min_holdout_records,
    )


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
    parser.add_argument(
        "--corpus", type=str, default=None,
        help=(
            "A curated dataset directory (build-corpus); defaults to the one "
            "--checkpoint recorded, if any. Fails fast when it has been "
            "rebuilt since training."
        ),
    )
    parser.add_argument("--abilities-root", type=str, default=DEFAULT_ABILITIES_ROOT)
    parser.add_argument(
        "--sealed-encoder-checkpoint", type=str,
        default="models/sealed/encoder/latest.pt",
        help="Read side by side with the effects cache in the decodability battery",
    )


def run_evaluate_effect_model(args: argparse.Namespace) -> int:
    from effects.application.evaluate_effect_model import (
        CorpusMismatchError,
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
        corpus=Path(args.corpus) if args.corpus else None,
        abilities_root=Path(args.abilities_root),
        sealed_encoder_checkpoint=Path(args.sealed_encoder_checkpoint),
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
    return 0 if report.ships else 1


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
