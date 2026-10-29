"""``collect-coverage``: put every card into a game, not just the popular ones.

Decks are built over the whole converted corpus, weighted toward the cards with
the fewest records, and played until every card is satisfied or retired.
Coverage matches write effect records only, because their decks are no fair
self-play sample; the guard is the Java worker's records-only mode, where the
writers are constructed.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.parallel_map import run_parallel
from effects.domain import default_paths
from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    CoverageUnit,
    DeckCandidate,
    TextCoverage,
    awaited_texts,
    deck_weights,
    held_out_texts_of,
    is_complete,
    is_deckable,
    qualifying_cards,
    rank_by_consult,
    retire_stalled,
    text_deck_weights,
)
from effects.domain.record_quality import record_refusal
from effects.domain.records import RecordKind, acting_keys

if TYPE_CHECKING:
    from effects.domain.provenance import ProvenanceKey
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.converted_trees import ConvertedTrees

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_DIR = Path(default_paths.DEFAULT_RECORDS_DIR)
DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = tuple(
    Path(folder) for folder in default_paths.DEFAULT_CARDS_FOLDERS
)
#: Java worker processes, for both collectors.
DEFAULT_WORKERS = 12
DEFAULT_TARGET_RECORDS = 50
DEFAULT_DECKS_PER_ROUND = 500
DEFAULT_NO_PROGRESS_ROUNDS = 3

#: Shards between progress lines while counting: the first pass reads the whole
#: existing corpus, and silence that long looks like a hang.
PROGRESS_EVERY_SHARDS = 10

#: A 40-card limited deck: 23 nonlands plus basics.
NONLANDS_PER_DECK = 23
DECK_SIZE = 40

#: Seeds the per-round deck sampler so which cards fill a round is reproducible.
RANDOM_SEED = 42

#: FR-036's floor: distinct games a held-out text must act in.
DEFAULT_MIN_TEXT_GAMES = 5


@dataclass
class CollectCoverageConfig:
    effect_records: Path = field(default_factory=lambda: DEFAULT_RECORDS_DIR)
    cards_folders: tuple[Path, ...] = DEFAULT_CARDS_FOLDERS
    split_from: Path | None = None
    exclude_cards: Path | None = None
    target_records: int = field(default=DEFAULT_TARGET_RECORDS, metadata={
        "help": "Per-card satisfaction goal (default: %(default)s)",
    })
    decks_per_round: int = field(default=DEFAULT_DECKS_PER_ROUND, metadata={
        "help": "Decks played as matches per round (default: %(default)s)",
    })
    no_progress_rounds: int = field(default=DEFAULT_NO_PROGRESS_ROUNDS, metadata={
        "help": (
            "Consecutive rounds without a new qualifying record before a card "
            "retires, which is what makes the run end (default: %(default)s)"
        ),
    })
    workers: int = field(default=DEFAULT_WORKERS, metadata={
        "help": "Parallel Java worker processes (default: %(default)s)",
    })
    caps: CollectionCaps = field(default_factory=CollectionCaps)
    #: ``--only-cards``: the ``holdout-cards`` list. Set, the run is the
    #: held-out coverage round (FR-035): decks from these cards alone, and the
    #: unit is the held-out ability text rather than the card.
    only_cards: Path | None = None
    min_text_games: int = field(default=DEFAULT_MIN_TEXT_GAMES, metadata={
        "help": (
            "With --only-cards: distinct games a held-out text must act in to "
            "be satisfied (FR-036; default: %(default)s)"
        ),
    })

    def converted_trees(self) -> ConvertedTrees:
        """The trees ``--cards-folder`` names. Deck candidates and the coverage
        unit come from the card tree alone (FR-046, FR-048): a token script
        could never be satisfied or retired."""
        from effects.infrastructure.converted_trees import ConvertedTrees

        return ConvertedTrees.from_folders(self.cards_folders)

    def holdout_source(self) -> HoldoutSource:
        return HoldoutSource(
            effect_records=self.effect_records, exclude_cards=self.exclude_cards,
            split_from=self.split_from,
        )

    def no_converted_cards(self) -> str:
        """The error for a run whose card tree holds no converted card."""
        where = self.converted_trees().cards_folder or ", ".join(
            str(folder) for folder in self.cards_folders
        )
        return (
            f"No converted cards under {where}. Run 'python -m price_predictor "
            "convert' first."
        )


def count_shard(shard: Path) -> Counter[str]:
    """One shard's qualifying records, keyed by folded card name to match the
    converted (lowercase) unit names. Top-level so a process pool can pickle it.
    """
    from effects.domain.card_names import fold_card_name
    from effects.infrastructure.record_io import read_shard

    counts: Counter[str] = Counter()
    for record in read_shard(shard):
        for name in qualifying_cards(record):
            counts[fold_card_name(name)] += 1
    return counts


class _ShardCounter:
    """Counts each shard once across a whole run, not once per round.

    Sound because a shard is final once counted: a round's workers are all
    terminated before ``play_round`` returns, and a shard's name carries its
    JVM lifetime, so no later round reopens it.
    """

    def __init__(self, progress_every: int = PROGRESS_EVERY_SHARDS) -> None:
        self._counted: set[Path] = set()
        self._progress_every = progress_every

    def _count_fresh[T](
        self,
        records_dir: Path,
        count: Callable[[Path], T],
        apply: Callable[[T], None],
        *,
        workers: int | None,
    ) -> int:
        """``count`` each shard not yet read, in parallel, and ``apply`` it;
        returns how many. Applying adds, so the order shards finish in cannot
        matter. ``count`` must pickle."""
        from effects.infrastructure.record_io import iter_shards

        fresh = [
            shard for shard in iter_shards(Path(records_dir))
            if shard not in self._counted
        ]
        for part in run_parallel(
            fresh, count, workers=workers, label="Counted",
            progress_every=self._progress_every,
        ):
            apply(part)
        self._counted.update(fresh)
        return len(fresh)


class CoverageCounter(_ShardCounter):
    """Each card's qualifying records, each shard counted once across the run.
    ``CardCoverage.records`` is accumulated, so nothing else may overwrite it.
    """

    def update(
        self,
        coverage: Mapping[str, CardCoverage],
        records_dir: Path,
        *,
        workers: int | None = None,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read."""

        def add(counts: Counter[str]) -> None:
            for name, records in counts.items():
                card = coverage.get(name)
                if card is not None:
                    card.records += records

        return self._count_fresh(records_dir, count_shard, add, workers=workers)


def build_coverage_decks(
    weights: dict[str, float],
    candidates: Mapping[str, DeckCandidate],
    count: int,
    *,
    rng: random.Random,
) -> list[list[str]]:
    """``count`` 40-card decks drawn from ``weights`` (FR-046).

    23 nonlands sampled with replacement in proportion to weight, since the
    point is to get a card into games rather than a deck anyone would play,
    then basics from ``compute_basic_lands``. A weighted card that is no
    candidate has no cost to read and is skipped.
    """
    from sealed.domain.manabase import compute_basic_lands

    pool = [(name, w) for name, w in weights.items() if name in candidates and w > 0]
    if not pool:
        return []
    names = [name for name, _ in pool]
    ws = [w for _, w in pool]

    decks: list[list[str]] = []
    for _ in range(count):
        nonlands = rng.choices(names, weights=ws, k=NONLANDS_PER_DECK)
        basics = compute_basic_lands([candidates[n] for n in nonlands])
        deck = list(nonlands)
        for land, n in basics.items():
            deck.extend([land] * n)
        decks.append(deck)
    return decks


@dataclass
class CoverageResidues:
    """What the run could not reach, split by why (FR-051): which cards need
    forcing by an intervention and which merely need more games."""

    uncastable: tuple[str, ...] = ()
    castable_but_short: tuple[str, ...] = ()

    def render(self, target: int) -> str:
        return (
            f"Residues: {len(self.uncastable)} cards the consult judged "
            f"uncastable, {len(self.castable_but_short)} castable cards short "
            f"of {target} records. Both fall to stage-three interventions."
        )


def residues(
    coverage: dict[str, CardCoverage], target: int,
) -> CoverageResidues:
    """Each unsatisfied card counted under its consult verdict."""
    uncastable: list[str] = []
    short: list[str] = []
    for name, card in sorted(coverage.items()):
        if card.satisfied(target):
            continue
        if card.verdict == ConsultVerdict.UNCASTABLE:
            uncastable.append(name)
        else:
            short.append(name)
    return CoverageResidues(tuple(uncastable), tuple(short))


TWO_HOLDOUT_SOURCES = (
    "pass one source of held-out cards, not both: --exclude-cards "
    "names the list this corpus is being depleted against, "
    "--split-from inherits a checkpoint's recorded split, and a "
    "disagreement between them is silent"
)


#: The name a depletion list takes inside the corpus it depleted.
HOLDOUT_LIST_NAME = "holdout-cards.txt"


@dataclass(frozen=True, slots=True)
class HoldoutSource:
    """Where a collector writes, and the one source of the cards it keeps out:
    ``--exclude-cards`` (needs no trained model) or ``--split-from`` (an
    existing checkpoint's split). Both at once is refused rather than merged,
    since a merge would hide a disagreement."""

    effect_records: Path = DEFAULT_RECORDS_DIR
    exclude_cards: Path | None = None
    split_from: Path | None = None

    def __post_init__(self) -> None:
        if self.split_from is not None and self.exclude_cards is not None:
            raise ValueError(TWO_HOLDOUT_SOURCES)

    @classmethod
    def from_flags(
        cls, *,
        effect_records: Path | None,
        exclude_cards: Path | None,
        split_from: Path | None,
        training_corpus: Path | None = None,
        only_cards: Path | None = None,
    ) -> HoldoutSource:
        """The source a collector's flags name, refused before any worker starts.

        ``--training-corpus DIR`` stands for ``--effect-records DIR`` with
        ``DIR/holdout-cards.txt`` as the depletion list, so a corpus and the
        list that depleted it cannot be named apart.

        Raises ``ValueError`` when ``--only-cards`` comes with a holdout to keep
        out, when ``--training-corpus`` comes with a flag it implies (an
        explicit default ``--effect-records`` included) or its directory holds
        no depletion list, or when two sources name the held-out cards.
        """
        if only_cards is not None:
            clashing = [
                flag for flag, value in (
                    ("--exclude-cards", exclude_cards),
                    ("--training-corpus", training_corpus),
                    ("--split-from", split_from),
                ) if value
            ]
            if clashing:
                # FR-038: opposite instructions, so one of them is a mistake.
                raise ValueError(
                    f"--only-cards decks the held-out cards; {', '.join(clashing)} "
                    "keeps a holdout out of every deck. Pass one or the other."
                )
        if training_corpus is None:
            return cls(
                effect_records=effect_records or DEFAULT_RECORDS_DIR,
                exclude_cards=exclude_cards, split_from=split_from,
            )
        if effect_records is not None or exclude_cards:
            raise ValueError(
                "pass one source of the corpus location, not both: "
                "--training-corpus DIR means --effect-records DIR with its own "
                f"{HOLDOUT_LIST_NAME}"
            )
        listed = training_corpus / HOLDOUT_LIST_NAME
        if not listed.is_file():
            raise ValueError(
                f"{training_corpus} holds no {HOLDOUT_LIST_NAME}. A training "
                "corpus is depleted against a list, and without it this run "
                "would deck the held-out cards and contaminate the split. Write "
                "one with 'python -m effects holdout-cards', or pass "
                "--effect-records and --exclude-cards separately if they "
                "genuinely live apart."
            )
        return cls(
            effect_records=training_corpus, exclude_cards=listed, split_from=split_from,
        )

    def exclusions(self) -> frozenset[str]:
        """Cards to keep out of every deck, folded, from whichever source names them."""
        if self.exclude_cards is not None:
            return read_name_list(self.exclude_cards)
        return load_held_out(self.split_from)


def read_name_list(path: Path) -> frozenset[str]:
    """The card names a one-per-line list holds, folded to match converted
    names; blank lines skipped."""
    from effects.domain.card_names import fold_card_name

    return frozenset(
        fold_card_name(line.strip())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_held_out(split_from: Path | None) -> frozenset[str]:
    """Cards to keep out of every deck (FR-045): the held-out cards of the
    checkpoint whose card-disjoint split this run must not contaminate."""
    if split_from is None:
        return frozenset()
    from effects.infrastructure.effect_model_store import load_effect_checkpoint

    provenance = load_effect_checkpoint(Path(split_from)).provenance
    return frozenset(provenance.held_out_cards)


def build_coverage_units(
    card_files: dict[str, str], held_out: frozenset[str],
) -> dict[str, CardCoverage]:
    """Every deckable card, minus the holdout. Takes the loaded ``name ->
    script`` mapping because ``run`` has already paid for reading the tree."""
    return {
        name: CardCoverage(name=name)
        for name in card_files
        if name not in held_out
    }


def non_deck_names(texts: Mapping[str, str]) -> set[str]:
    """The cards whose converted text names a type no deck holds."""
    return {name for name, text in texts.items() if not is_deckable(text)}


@contextmanager
def collector_supervisor(
    *,
    workers: int,
    effect_records: Path,
    caps: CollectionCaps,
    variant_scripts: Path | None = None,
) -> Iterator[CollectorSupervisor]:
    """A records-only supervisor that is stopped however its rounds end:
    without ``stop()`` the worker logs holding the latched failure reports
    stay unflushed until the interpreter exits."""
    from effects.infrastructure.collector_connector import CollectorSupervisor

    supervisor = CollectorSupervisor(
        worker_count=workers, effect_records=effect_records, caps=caps,
        variant_scripts=variant_scripts,
    )
    try:
        yield supervisor
    finally:
        supervisor.stop()


def play_decks(
    supervisor: CollectorSupervisor,
    decks: list[list[str]],
    decks_file: Path,
    *,
    label: str,
    matches: int,
) -> None:
    """Play one round of ``decks``, which a records-only round reads from a file."""
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file

    write_deck_file(decks, decks_file, label=label, set_code=COVERAGE_SET_CODE)
    supervisor.play_round(decks_file, matches=matches)


@dataclass(frozen=True)
class CoverageRounds:
    """What the round loop needs from one kind of coverage unit: the card run
    and the held-out round differ in what a unit is, how its cards are weighted
    and which units have to finish."""

    units: Mapping[str, CoverageUnit]
    #: The units whose completion ends the run: every card, but only the
    #: held-out texts some castable card carries (FR-037).
    awaited: Mapping[str, CoverageUnit]
    target: int
    candidates: Mapping[str, DeckCandidate]
    #: The next round's per-card deck weights, the consult applied.
    weights: Callable[[], dict[str, float]]
    #: The units a round attempted, from the cards its decks held.
    units_decked: Callable[[set[str]], set[str]]
    #: Counts the shards the last round wrote into ``units``; returns how many.
    count_new_shards: Callable[[], int]
    #: Logged when the weighted cards yield no deck at all.
    no_decks: str


def round_summary(
    round_number: int, rounds: CoverageRounds, retired: list[str],
) -> str:
    satisfied = sum(
        1 for unit in rounds.units.values() if unit.satisfied(rounds.target)
    )
    remaining = sum(
        1 for unit in rounds.awaited.values() if not unit.done(rounds.target)
    )
    return (
        f"round {round_number}: {satisfied} satisfied, {remaining} remaining, "
        f"{len(retired)} retired this round"
    )


def play_rounds(
    rounds: CoverageRounds,
    supervisor: CollectorSupervisor,
    config: CollectCoverageConfig,
) -> bool:
    """Play rounds until every awaited unit is done. True when Ctrl-C ended them."""
    decks_file = Path(config.effect_records) / "coverage-decks.txt"
    rng = random.Random(RANDOM_SEED)
    round_number = 0
    while not is_complete(rounds.awaited, rounds.target):
        round_number += 1
        previous = {key: unit.records for key, unit in rounds.units.items()}
        weights = rounds.weights()
        if not weights:
            break
        decks = build_coverage_decks(
            weights, rounds.candidates, config.decks_per_round, rng=rng,
        )
        if not decks:
            logger.error("%s", rounds.no_decks)
            break
        decked = rounds.units_decked({name for deck in decks for name in deck})
        play_decks(
            supervisor, decks, decks_file,
            label="coverage", matches=config.decks_per_round,
        )
        read = rounds.count_new_shards()
        logger.info("Counted %d new shard(s) from round %d", read, round_number)
        retired = retire_stalled(
            rounds.units, previous, no_progress_rounds=config.no_progress_rounds,
            decked=decked,
        )
        logger.info("%s", round_summary(round_number, rounds, retired))
        if supervisor.interrupted:
            # ForgeWorkerPool swallows SIGINT, so without this Ctrl-C would only
            # skip to the next round.
            logger.warning(
                "Interrupted after round %d; stopping instead of starting "
                "another round", round_number,
            )
            return True
    return False


def run(config: CollectCoverageConfig) -> int:
    """Play rounds until every card is satisfied or retired."""
    if config.only_cards is not None:
        return run_text_coverage(config)

    trees = config.converted_trees()
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", config.no_converted_cards())
        return 1

    held_out = config.holdout_source().exclusions()
    texts = {name: card.text for name, card in converted.items()}
    non_deck = non_deck_names(texts)
    coverage = build_coverage_units(
        {
            name: card.script_file for name, card in converted.items()
            if name not in non_deck
        },
        held_out,
    )
    if not coverage:
        logger.error("No deckable cards under %s", trees.cards_folder)
        return 1
    logger.info(
        "Coverage unit: %d cards (%d held out, %d of a non-deck type), target "
        "%d records each",
        len(coverage), len(held_out), len(non_deck), config.target_records,
    )

    records_dir = Path(config.effect_records)
    counter = CoverageCounter()
    seeded = counter.update(coverage, records_dir)
    logger.info(
        "Counted %d existing shard(s); %d of %d cards already at target",
        seeded,
        sum(1 for c in coverage.values() if c.satisfied(config.target_records)),
        len(coverage),
    )

    verdicts = consult_castability(sorted(coverage))
    for name, verdict in verdicts.items():
        if name in coverage:
            coverage[name].verdict = verdict

    rounds = CoverageRounds(
        units=coverage,
        awaited=coverage,
        target=config.target_records,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items()
        },
        weights=lambda: rank_by_consult(
            deck_weights(coverage, config.target_records), verdicts,
        ),
        units_decked=lambda cards: cards,
        count_new_shards=lambda: counter.update(coverage, records_dir),
        no_decks="No card with a weight has converted text; nothing to deck.",
    )
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records,
        caps=config.caps,
    ) as supervisor:
        interrupted = play_rounds(rounds, supervisor, config)

    report = residues(coverage, config.target_records)
    logger.info("%s", report.render(config.target_records))
    # Residues print either way, but an interrupted run did not finish.
    return 130 if interrupted else 0


def consult_castability(names: list[str]) -> dict[str, ConsultVerdict]:
    """Ask a live Forge whether each card can be cast at all (FR-047). A failed
    consult leaves every card ``unknown``, ranked as castable: the safe
    direction, since the consult only ranks."""
    from effects.infrastructure.one_shot_workers import castability_verdicts

    try:
        verdicts = castability_verdicts(names)
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Castability consult unavailable (%s); every card ranks as "
            "castable, which only slows the run down", exc,
        )
        return {}
    return {name: ConsultVerdict.parse(raw) for name, raw in verdicts.items()}


# ── the held-out coverage round (FR-035–FR-038) ─────────────────────────


def count_text_shard(
    shard: Path, *, roots: dict[str, str], texts: frozenset[str],
) -> dict[str, set[str]]:
    """``held-out text -> the game ids it resolved in``, over one shard.

    Counted as build-corpus counts a held-out text's games (FR-053): from the
    resolution records its refusal rule keeps, through the first acting key
    whose text is held out. Top-level so a process pool can pickle it; each
    process resolves keys through its own :func:`_script_texts`, because a
    sidecar cache does not pickle.
    """
    from effects.infrastructure.record_io import read_shard

    script_texts = _script_texts(tuple(sorted(roots.items())))
    games: dict[str, set[str]] = defaultdict(set)
    for record in read_shard(shard):
        if record.kind is not RecordKind.RESOLUTION:
            continue
        if record_refusal(record, script_texts.sidecars.resolution_of) is not None:
            continue
        text = next(
            (text for key in acting_keys(record) if (text := script_texts.of(key)) in texts),
            None,
        )
        if text is not None:
            games[text].add(record.game_id)
    return dict(games)


class _ScriptTexts:
    """Each provenance key's normalized script text, resolved once per key."""

    def __init__(self, roots: tuple[tuple[str, str], ...]) -> None:
        from effects.infrastructure.sidecar_io import SidecarCache

        self.sidecars = SidecarCache({tree: Path(root) for tree, root in roots})
        self._by_key: dict[ProvenanceKey, str | None] = {}

    def of(self, key: ProvenanceKey) -> str | None:
        from effects.domain.text_holdout import normalize_script_text

        if key not in self._by_key:
            line = self.sidecars.line_or_none(key)
            self._by_key[key] = (
                normalize_script_text(line.script_text)
                if line is not None and line.script_text else None
            )
        return self._by_key[key]


@lru_cache(maxsize=1)
def _script_texts(roots: tuple[tuple[str, str], ...]) -> _ScriptTexts:
    """The process's resolver for ``roots``, kept across the shards it counts,
    which name largely the same cards."""
    return _ScriptTexts(roots)


class TextCoverageCounter(_ShardCounter):
    """Each held-out text's games, each shard counted once across the run."""

    def __init__(
        self, roots: dict[str, str], progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__(progress_every)
        self._roots = dict(roots)

    def update(
        self, coverage: Mapping[str, TextCoverage], records_dir: Path,
        *, workers: int | None = None,
    ) -> int:
        """Add the games of shards not yet read. Returns how many it read."""

        def add(games_by_text: dict[str, set[str]]) -> None:
            for text, games in games_by_text.items():
                coverage[text].games |= games

        count = partial(count_text_shard, roots=self._roots, texts=frozenset(coverage))
        return self._count_fresh(records_dir, count, add, workers=workers)


@dataclass
class TextResidues:
    """What the held-out round could not reach (FR-037, SC-004)."""

    under_floor: dict[str, int] = field(default_factory=dict)
    uncastable: tuple[str, ...] = ()

    def render(self, floor: int) -> str:
        lines = [
            f"{len(self.under_floor)} held-out text(s) with a castable carrier "
            f"under {floor} games; {len(self.uncastable)} with no castable "
            "carrier, never counted as satisfied."
        ]
        lines += [f"  under floor ({games} games): {text}"
                  for text, games in sorted(self.under_floor.items())]
        lines += [f"  uncastable: {text}" for text in self.uncastable]
        return "\n".join(lines)


def text_residues(coverage: dict[str, TextCoverage], floor: int) -> TextResidues:
    under = {
        text: len(unit.games) for text, unit in coverage.items()
        if unit.castable and not unit.satisfied(floor)
    }
    uncastable = tuple(sorted(
        text for text, unit in coverage.items() if not unit.castable
    ))
    return TextResidues(under, uncastable)


def held_out_round_caps(caps: CollectionCaps) -> CollectionCaps:
    """Full strength (FR-035): every record the held-out cards produce is kept,
    because these are the only games that will ever see them.
    """
    return replace(caps, playability_rate=1.0)


def run_text_coverage(config: CollectCoverageConfig) -> int:
    """The held-out coverage round: listed cards only, keyed by held-out text."""

    from effects.domain.card_names import fold_card_name

    trees = config.converted_trees()
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", config.no_converted_cards())
        return 1
    listed = read_name_list(config.only_cards)
    card_files = trees.card_files
    texts_by_card = trees.card_texts()
    listed_files = {
        name: script for name, script in card_files.items()
        if fold_card_name(name) in listed
    }
    texts = {name: converted[name].text for name in listed_files}
    # Held-out texts are decided over every listed card, non-deck ones
    # included, or those would read as unlisted carriers and take their texts
    # out of the round; only then do non-deck cards leave as carriers.
    non_deck = non_deck_names(texts)
    present = frozenset(name for name in listed_files if name not in non_deck)
    carriers = {
        text: cards - non_deck
        for text, cards in held_out_texts_of(
            frozenset(listed_files), texts_by_card,
        ).items()
        if cards - non_deck
    }
    if non_deck:
        logger.info(
            "Held-out round: %d listed card(s) of a non-deck type left out",
            len(non_deck),
        )
    if not carriers:
        logger.error(
            "None of the %d card(s) in %s carries a text only listed cards "
            "carry; nothing to cover.", len(listed), config.only_cards,
        )
        return 1

    verdicts = consult_castability(sorted(present))
    coverage = {
        text: TextCoverage(text=text, castable=any(
            verdicts.get(card, ConsultVerdict.UNKNOWN) != ConsultVerdict.UNCASTABLE
            for card in cards
        ))
        for text, cards in carriers.items()
    }
    floor = config.min_text_games
    records_dir = Path(config.effect_records)
    counter = TextCoverageCounter(trees.root_paths)
    counter.update(coverage, records_dir)
    logger.info(
        "Held-out round: %d text(s) on %d card(s); %d already at %d games; %d "
        "with no castable carrier",
        len(coverage), len(present),
        sum(1 for unit in coverage.values() if unit.satisfied(floor)), floor,
        sum(1 for unit in coverage.values() if not unit.castable),
    )

    rounds = CoverageRounds(
        units=coverage,
        awaited=awaited_texts(coverage),
        target=floor,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items() if name in present
        },
        weights=lambda: rank_by_consult(
            text_deck_weights(coverage, carriers, floor), verdicts,
        ),
        units_decked=lambda cards: {
            text for text, carrying in carriers.items() if carrying & cards
        },
        count_new_shards=lambda: counter.update(coverage, records_dir),
        no_decks="No unfinished held-out text has a card with converted text.",
    )
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records,
        caps=held_out_round_caps(config.caps),
    ) as supervisor:
        interrupted = play_rounds(rounds, supervisor, config)

    logger.info("%s", text_residues(coverage, floor).render(floor))
    return 130 if interrupted else 0
