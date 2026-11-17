"""``collect-coverage``: put every card into a game, not just the popular ones.

Decks are built over the whole converted corpus, weighted toward the cards with
the fewest records, and played until every card is satisfied or retired.
Coverage matches write effect records only, because their decks are no fair
self-play sample; the guard is the Java worker's records-only mode, where the
writers are constructed.

``--wording-cards`` narrows the card run to a list (FR-038): the cards with an
activated "any target" damage ability, say, so the wording reaches the corpus
in games of its own rather than at the rate the whole corpus would deck it.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from effects.application.collector_common import (
    CollectorConfig,
    build_coverage_decks,
    collector_supervisor,
    positive_int,
    read_name_list,
    record_damaged_shards,
)
from effects.application.parallel_map import run_parallel
from effects.domain.card_names import fold_card_name
from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import (
    HELD_OUT_GAME_FLOOR,
    CardCoverage,
    CardOfScript,
    ConsultVerdict,
    CoverageUnit,
    DeckCandidate,
    TextCoverage,
    awaited_texts,
    cards_shown,
    deck_weights,
    held_out_texts_of,
    is_complete,
    is_deckable,
    qualifying_cards,
    rank_by_consult,
    retire_stalled,
    text_deck_weights,
)
from effects.domain.game_ends import GameEnds
from effects.domain.provenance import SidecarLookup, SourceTree, split_script_file
from effects.domain.record_quality import quality_defect, record_refusal
from effects.domain.records import EffectRecord, RecordKind, acting_keys

if TYPE_CHECKING:
    from effects.domain.split import HeldOutCards
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.converted_trees import ConvertedTrees

logger = logging.getLogger(__name__)

DEFAULT_TARGET_RECORDS = 50
DEFAULT_NO_PROGRESS_ROUNDS = 3

#: Shards between progress lines while counting: the first pass reads the whole
#: existing corpus, and silence that long looks like a hang.
PROGRESS_EVERY_SHARDS = 10


WORDING_BESIDE_ONLY_CARDS = (
    "pass --wording-cards or --only-cards, not both: one is a card round over "
    "a list, the other the held-out round counted per text"
)


@dataclass
class CollectCoverageConfig(CollectorConfig):
    target_records: int = field(default=DEFAULT_TARGET_RECORDS, metadata={
        "type": positive_int,
        "help": "Per-card satisfaction goal (default: %(default)s)",
    })
    no_progress_rounds: int = field(default=DEFAULT_NO_PROGRESS_ROUNDS, metadata={
        "type": positive_int,
        "help": (
            "Consecutive rounds without a new qualifying record before a card "
            "retires, which is what makes the run end (default: %(default)s)"
        ),
    })
    min_text_games: int = field(default=HELD_OUT_GAME_FLOOR, metadata={
        "type": positive_int,
        "help": (
            "With --only-cards: distinct games a held-out text must act in to "
            "be satisfied (FR-036; default: %(default)s)"
        ),
    })
    wording_cards: Path | None = field(default=None, metadata={
        "help": (
            "A coverage round over these cards alone, one card name per line "
            "(FR-038): the coverage unit and the decks, the holdout still kept "
            "out. Refused beside --only-cards"
        ),
    })
    #: Earlier runs' record directories whose coverage counts toward
    #: satisfaction (``--count-from``, repeatable); read once, never written.
    count_from: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.wording_cards is not None and self.deck_holdout.only_cards is not None:
            raise ValueError(WORDING_BESIDE_ONLY_CARDS)
        self.count_from = tuple(Path(directory) for directory in self.count_from or ())
        error = count_from_error(self.effect_records, self.count_from)
        if error is not None:
            raise ValueError(error)


def count_from_error(effect_records: Path, count_from: tuple[Path, ...]) -> str | None:
    """Why ``--count-from`` cannot be read beside ``--effect-records``, or None.

    Shards are found recursively, so a directory holding the run's own, or
    held by it, or two overlapping ones, would count a shard twice."""
    directories = [Path(effect_records), *count_from]
    resolved = [directory.resolve() for directory in directories]
    for index, directory in enumerate(resolved):
        for other_index, other in enumerate(resolved[:index]):
            if directory == other or directory.is_relative_to(other) or (
                other.is_relative_to(directory)
            ):
                first = "--effect-records" if other_index == 0 else "--count-from"
                return (
                    f"--count-from {directories[index]} overlaps {first} "
                    f"{directories[other_index]}: their shards would count twice"
                )
    for directory in count_from:
        if not Path(directory).is_dir():
            return f"--count-from {directory}: no such directory"
    return None



def no_converted_cards(trees: ConvertedTrees) -> str:
    """The error for a run whose card tree holds no converted card. Deck
    candidates and the coverage unit come from the card tree alone (FR-046,
    FR-048): a token script could never be satisfied or retired."""
    where = trees.cards_folder or ", ".join(str(folder) for folder in trees.roots.values())
    return (
        f"No converted cards under {where}. Run 'python -m price_predictor "
        "convert' first."
    )


@dataclass(frozen=True)
class ShardCoverage:
    """What one shard adds to the card run: each card's credited records,
    every card its games showed (a round charges a stalled round to those
    alone), and how many records it holds."""

    credits: Counter[str]
    seen: frozenset[str]
    records: int = 0
    #: What ended the shard's read early (a damaged member or line), its
    #: records before the damage counted; None for a whole shard.
    damaged: str | None = None
    #: Games without a game-end marker, whose records credit nothing.
    cut_games: int = 0


def _card_of_script(sidecars: SidecarLookup | None) -> CardOfScript | None:
    """``script file -> converted card name`` (a sidecar's ``card``), for
    card-tree scripts only; None without sidecars."""
    if sidecars is None:
        return None

    def card_of(script_file: str) -> str | None:
        if split_script_file(script_file)[0] != SourceTree.CARDSFOLDER:
            return None
        try:
            return sidecars.get(script_file).card
        except (KeyError, OSError, ValueError):
            return None

    return card_of


def _kept(record: EffectRecord, sidecars: SidecarLookup | None) -> bool:
    """Whether build-corpus would keep the record: not a ``turn_action``, and
    not refused (its quality rule alone without sidecars)."""
    if record.kind is RecordKind.TURN_ACTION:
        return False
    if sidecars is None:
        return quality_defect(record) is None
    return record_refusal(record, sidecars.resolution_of) is None


def count_shard(shard: Path, *, roots: Mapping[str, Path] | None = None) -> ShardCoverage:
    """One shard's qualifying records, keyed by folded card name to match the
    converted (lowercase) unit names. Top-level so a process pool can pickle it.

    Only the records build-corpus keeps count (python D1): no
    ``turn_action`` (a draw or an untap is no record of the card), none it
    refuses, and none of a game its shard holds no game-end marker for (a
    cut game, which build-corpus drops whole; credits are held per game
    until the shard is read). A record of an older generation stops the
    count with :class:`~effects.domain.shard_generation.OlderGenerationError`,
    as it stops every reader: it is no damage to count past. Under ``roots`` an entity counts as the
    converted card its printed script is (python D6), and a static's host
    through its script (python D5). The cards shown are every game's, a cut
    one's too: its games showed them.
    """
    from effects.infrastructure.converted_trees import process_sidecars
    from effects.infrastructure.shard_io import SHARD_DAMAGE, read_shard

    sidecars = process_sidecars(roots) if roots else None
    card_of = _card_of_script(sidecars)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    ends = GameEnds()
    records = 0
    damaged = None
    try:
        for record in read_shard(shard):
            records += 1
            ends.note(record)
            seen.update(fold_card_name(name) for name in cards_shown(record, card_of))
            if not _kept(record, sidecars):
                continue
            for name in qualifying_cards(record, card_of):
                by_game[record.game_id][fold_card_name(name)] += 1
    except SHARD_DAMAGE as exc:
        damaged = str(exc)
    cut = ends.cut
    counts: Counter[str] = Counter()
    for game, credits in by_game.items():
        if game not in cut:
            counts.update(credits)
    return ShardCoverage(counts, frozenset(seen), records, damaged, len(cut))


class _ShardCounter:
    """Counts each shard once across a whole run, not once per round.

    Sound because a shard is final once counted: a round's workers have all
    ended before ``play_round`` returns (each finished its game when asked to
    stop, or was killed past the stop timeout; a worker spawned while the
    pool shuts down is killed by the pool too), and a shard's name carries
    its JVM lifetime, so no later round reopens it. A worker the kill could
    not end is the exception: the supervisor names it, and its shard is
    counted as it stood when read, its game in progress cut.
    """

    def __init__(self, progress_every: int = PROGRESS_EVERY_SHARDS) -> None:
        self._counted: set[Path] = set()
        #: Gen-3 games without a game-end marker the counts left out: over
        #: the whole run, and in the last ``update``.
        self.cut_games = 0
        self.last_cut = 0
        #: Every damaged shard read so far, as its read's message: counted up
        #: to the damage, once, like any other shard.
        self.damaged: list[str] = []
        self._progress_every = progress_every
        #: Every card the shards of the last ``update`` showed.
        self.seen: set[str] = set()
        #: Records per shard counted so far.
        self._records: dict[Path, int] = {}

    def records_of(self, run_id: str) -> int:
        """Records in the counted shards of run ``run_id`` (``{run_id}.*``)."""
        prefix = f"{run_id}."
        return sum(n for shard, n in self._records.items() if shard.name.startswith(prefix))

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
        from effects.infrastructure.shard_io import iter_shards

        fresh = [
            shard for shard in iter_shards(records_dir)
            if shard not in self._counted
        ]
        self.last_cut = 0
        for shard, part in zip(fresh, run_parallel(
            fresh, count, workers=workers, label="Counted",
            progress_every=self._progress_every,
        ), strict=True):
            apply(part)
            self._records[shard] = part.records
            self.last_cut += part.cut_games
            if part.damaged is not None:
                logger.warning("%s; its %d records before the damage are counted, and "
                               "the shard is not read again", part.damaged, part.records)
                self.damaged.append(part.damaged)
        self._counted.update(fresh)
        self.cut_games += self.last_cut
        return len(fresh)


class CoverageCounter(_ShardCounter):
    """Each card's qualifying records, each shard counted once across the run.
    ``CardCoverage.records`` is accumulated, so nothing else may overwrite it.

    ``roots`` (the converted trees' sidecar roots) apply build-corpus's whole
    refusal rule and credit each entity as its converted card; without them
    only the quality rule applies and an entity counts by its name.
    """

    def __init__(
        self, roots: Mapping[str, Path] | None = None,
        progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__(progress_every)
        self._roots = dict(roots) if roots else None

    def update(
        self,
        coverage: Mapping[str, CardCoverage],
        records_dir: Path,
        *,
        workers: int | None = None,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read."""
        self.seen = set()

        def add(part: ShardCoverage) -> None:
            self.seen |= part.seen
            for name, records in part.credits.items():
                card = coverage.get(name)
                if card is not None:
                    card.records += records

        count = partial(count_shard, roots=self._roots)
        return self._count_fresh(records_dir, count, add, workers=workers)


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


def build_coverage_units(
    names: Iterable[str], held_out: frozenset[str],
) -> dict[str, CardCoverage]:
    """Every deckable card, minus the holdout."""
    return {
        name: CardCoverage(name=name)
        for name in names
        if name not in held_out
    }


def non_deck_names(texts: Mapping[str, str]) -> set[str]:
    """The cards whose converted text names a type no deck holds."""
    return {name for name, text in texts.items() if not is_deckable(text)}


@dataclass(frozen=True)
class Unresolved:
    """The cards Forge's card database does not hold, out of the pool the
    consult was asked about (P1 #11)."""

    cards: tuple[str, ...] = ()
    pool: int = 0

    @property
    def share(self) -> float:
        return len(self.cards) / self.pool if self.pool else 0.0

    def fields(self) -> dict:
        """What ``run.json`` records: the count, the share, and some names."""
        return {
            "unresolved_cards": len(self.cards),
            "unresolved_share": round(self.share, 4),
            "unresolved_examples": list(self.cards[:20]),
        }


@dataclass(kw_only=True)
class CoverageRounds(ABC):
    """What the round loop needs from one kind of coverage unit, and what both
    kinds hold: the units, the consult's verdicts, the counter that reads each
    round's shards into the units, and the cards a deck can be drawn from. The
    card run and the held-out round differ in what a unit is, how its cards
    are weighted and which units have to finish."""

    coverage: dict[str, CoverageUnit]
    verdicts: Mapping[str, ConsultVerdict]
    counter: CoverageCounter | TextCoverageCounter
    records_dir: Path
    target: int
    candidates: dict[str, DeckCandidate]
    #: Processes counting shards: the run's ``--workers``, never the CPU count.
    workers: int | None = None
    #: The cards left out because Forge's card database does not hold them.
    unresolved: Unresolved = field(default_factory=Unresolved)
    #: Logged when the weighted cards yield no deck at all.
    no_decks: ClassVar[str]

    @property
    def units(self) -> dict[str, CoverageUnit]:
        return self.coverage

    @property
    @abstractmethod
    def awaited(self) -> Mapping[str, CoverageUnit]:
        """The units whose completion ends the run: every card, but only the
        held-out texts some castable card carries (FR-037)."""

    @abstractmethod
    def weights(self) -> dict[str, float]:
        """The next round's per-card deck weights, the consult applied."""

    @abstractmethod
    def units_decked(self, cards: set[str]) -> set[str]:
        """The units a round attempted, from the cards its decks held."""

    def count_new_shards(self) -> int:
        """Count the shards the last round wrote into ``units``; how many."""
        return self.counter.update(self.coverage, self.records_dir, workers=self.workers)

    @abstractmethod
    def report(self) -> str:
        """What the rounds could not reach, printed however they ended."""


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


@dataclass(frozen=True)
class RoundsPlayed:
    """How a run's rounds ended: how many were played, and whether Ctrl-C
    ended them."""

    rounds: int
    interrupted: bool


def play_rounds(
    rounds: CoverageRounds,
    supervisor: CollectorSupervisor,
    config: CollectCoverageConfig,
) -> RoundsPlayed:
    """Play rounds until every awaited unit is done.

    The decks file is the run's own (``decks/coverage-decks.<run_id>.txt``):
    every worker reloads it at its start, so two runs on one records
    directory would otherwise play each other's decks.
    """
    decks_file = supervisor.decks_path("coverage")
    rng = random.Random(config.seed)
    round_number = 0
    while not is_complete(rounds.awaited, rounds.target):
        previous = {key: unit.progress for key, unit in rounds.units.items()}
        weights = rounds.weights()
        if not weights:
            break
        decks = build_coverage_decks(
            weights, rounds.candidates, config.decks_per_round, rng=rng,
        )
        if not decks:
            logger.error("%s", rounds.no_decks)
            break
        round_number += 1
        written = {name for deck in decks for name in deck}
        supervisor.play_round(
            decks, decks_file, label="coverage", matches=config.decks_per_round,
        )
        read = rounds.count_new_shards()
        logger.info("Counted %d new shard(s) from round %d", read, round_number)
        record_damaged_shards(supervisor, rounds.counter.damaged)
        supervisor.count_cut_games(rounds.counter.last_cut)
        # Charged to what the round's games showed: a worker draws its decks
        # with replacement, so some written decks are never played.
        retired = retire_stalled(
            rounds.units, previous, no_progress_rounds=config.no_progress_rounds,
            decked=rounds.units_decked(written & rounds.counter.seen),
            written=rounds.units_decked(written),
        )
        logger.info("%s", round_summary(round_number, rounds, retired))
        if supervisor.interrupted:
            # ForgeWorkerPool swallows SIGINT, so without this Ctrl-C would only
            # skip to the next round.
            logger.warning(
                "Interrupted after round %d; stopping instead of starting "
                "another round", round_number,
            )
            return RoundsPlayed(round_number, interrupted=True)
    return RoundsPlayed(round_number, interrupted=False)


def _play_to_completion(
    rounds: CoverageRounds, config: CollectCoverageConfig, caps: CollectionCaps,
) -> int:
    """Play ``rounds`` under a supervisor and report what they left; 130 when
    Ctrl-C ended them, 1 when they played and wrote no record at all."""
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records, caps=caps,
        stop=config.worker_stop,
    ) as supervisor:
        if rounds.unresolved.pool:
            supervisor.record(**rounds.unresolved.fields())
        record_damaged_shards(supervisor, rounds.counter.damaged)
        supervisor.count_cut_games(rounds.counter.cut_games)
        played = play_rounds(rounds, supervisor, config)
    logger.info("%s", rounds.report())
    # Residues print either way, but an interrupted run did not finish.
    if played.interrupted:
        return 130
    if not played.rounds:
        if is_complete(rounds.awaited, rounds.target):
            return 0
        logger.error(
            "No round was played and %d unit(s) are still unfinished: none of "
            "them could be decked.",
            sum(1 for unit in rounds.awaited.values() if not unit.done(rounds.target)),
        )
        return 1
    written = rounds.counter.records_of(supervisor.run_id)
    if written <= 0:
        logger.error(
            "%d round(s) played and no new effect record written under %s. "
            "The worker logs say why (under logs/, summed above): every match "
            "failing, or every deck naming a card Forge's card database does "
            "not hold.", played.rounds, config.effect_records,
        )
        return 1
    logger.info("Collected %d new effect records over %d round(s)", written, played.rounds)
    return 0


@dataclass(kw_only=True)
class CardRounds(CoverageRounds):
    """The card run's rounds: every deckable card a unit, and every one awaited."""

    coverage: dict[str, CardCoverage]
    counter: CoverageCounter
    no_decks: ClassVar[str] = "No card with a weight has converted text; nothing to deck."

    @property
    def awaited(self) -> dict[str, CardCoverage]:
        return self.coverage

    def weights(self) -> dict[str, float]:
        return rank_by_consult(deck_weights(self.coverage, self.target), self.verdicts)

    def units_decked(self, cards: set[str]) -> set[str]:
        return cards

    def report(self) -> str:
        return residues(self.coverage, self.target).render(self.target)


def run(config: CollectCoverageConfig) -> int:
    """Play rounds until every card is satisfied or retired."""
    # ``--only-cards``, the ``holdout-cards`` list: the held-out coverage
    # round (FR-035), decks from these cards alone, each unit a held-out
    # ability text rather than a card.
    if config.deck_holdout.only_cards is not None:
        return run_text_coverage(config)

    trees = config.source_trees.converted()
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", no_converted_cards(trees))
        return 1

    if config.wording_cards is not None:
        converted = wording_cards_of(converted, config.wording_cards)
        if not converted:
            logger.error(
                "None of the cards in %s has converted text under %s",
                config.wording_cards, trees.cards_folder,
            )
            return 1

    held_out = config.deck_holdout.exclusions()
    texts = {name: card.text for name, card in converted.items()}
    non_deck = non_deck_names(texts)
    coverage = build_coverage_units(
        (name for name in converted if name not in non_deck), held_out,
    )
    if not coverage:
        logger.error("No deckable cards under %s", trees.cards_folder)
        return 1
    logger.info(
        "Coverage unit: %d cards (%d held out, %d of a non-deck type), target "
        "%d records each",
        len(coverage), len(held_out), len(non_deck), config.target_records,
    )

    records_dir = config.effect_records
    counter = CoverageCounter(config.source_trees.sidecar_roots())
    seeded = _count_existing(counter, coverage, config)
    logger.info(
        "Counted %d existing shard(s); %d of %d cards already at target",
        seeded,
        sum(1 for c in coverage.values() if c.satisfied(config.target_records)),
        len(coverage),
    )

    verdicts = consult_castability(sorted(coverage))
    unresolved = unresolvable(coverage, verdicts)
    for name in unresolved.cards:
        del coverage[name]
    if not coverage:
        logger.error("No card under %s is in Forge's card database", trees.cards_folder)
        return 1
    for name, verdict in verdicts.items():
        if name in coverage:
            coverage[name].verdict = verdict

    rounds = CardRounds(
        coverage=coverage, verdicts=verdicts, counter=counter, unresolved=unresolved,
        records_dir=records_dir, target=config.target_records, workers=config.workers,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items()
        },
    )
    return _play_to_completion(rounds, config, config.caps)


def _count_existing(
    counter: CoverageCounter | TextCoverageCounter, coverage: Mapping,
    config: CollectCoverageConfig,
) -> int:
    """Count what the run's own directory and every ``--count-from`` directory
    already hold; how many shards. The ``--count-from`` directories are read
    here once and never again: the rounds write only to the run's own."""
    seeded = counter.update(coverage, config.effect_records, workers=config.workers)
    for directory in config.count_from:
        found = counter.update(coverage, directory, workers=config.workers)
        logger.info("Counted %d shard(s) from --count-from %s", found, directory)
        seeded += found
    return seeded


def unresolvable(units: Iterable[str], verdicts: Mapping[str, ConsultVerdict]) -> Unresolved:
    """The cards the consult left unanswered, reported; none when it answered
    nothing at all (a consult that could not run proves nothing).

    The castability verdict only ranks and never drops a card. This is a
    different rule: the consult answers every name Forge's card database
    holds, and a decks-only round refuses a whole deck naming any other card,
    so a card left unanswered cannot be decked at all. Kept, it would be
    unrecorded and so weighted the most, failing every match whose deck holds
    it until it retired unseen (P1 #11); it leaves the unit and the decks
    instead. No share stops the run; the report says how many, out of how
    many, and what a large share means."""
    pool = sorted(units)
    if not verdicts:
        return Unresolved(pool=len(pool))
    missing = tuple(name for name in pool if name not in verdicts)
    result = Unresolved(missing, len(pool))
    if missing:
        logger.warning(
            "%d of %d card(s) (%.1f%%) are not in Forge's card database (the "
            "castability consult left them unanswered) and are left out of the "
            "unit and the decks. A large share means ../forge is older than the "
            "converted cards: rebuild Forge or reconvert. %s",
            len(missing), len(pool), 100.0 * result.share,
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )
    return result


def wording_cards_of[C](converted: Mapping[str, C], path: Path) -> dict[str, C]:
    """The converted cards ``path`` lists, matched by folded name; the listed
    names with no converted card are logged, since a typo would otherwise
    shrink the round without a word."""
    listed = read_name_list(path)
    kept = {
        name: card for name, card in converted.items() if fold_card_name(name) in listed
    }
    missing = listed - {fold_card_name(name) for name in kept}
    logger.info(
        "Wording round: %d of %d listed card(s) have converted text", len(kept), len(listed),
    )
    if missing:
        logger.warning(
            "Listed in %s with no converted card: %s", path, ", ".join(sorted(missing)),
        )
    return kept


def consult_castability(names: list[str]) -> dict[str, ConsultVerdict]:
    """Ask a live Forge whether each card can be cast at all (FR-047). A failed
    consult leaves every card ``unknown``, ranked as castable: the safe
    direction, since the consult only ranks."""
    from effects.infrastructure.one_shot_workers import castability_verdicts

    try:
        verdicts = castability_verdicts(names)
    except OSError as exc:
        logger.warning(
            "Castability consult unavailable (%s); every card ranks as "
            "castable, which only slows the run down", exc,
        )
        return {}
    return {name: ConsultVerdict.parse(raw) for name, raw in verdicts.items()}


# ── the held-out coverage round (FR-035–FR-038) ─────────────────────────


@dataclass(frozen=True)
class TextShardCoverage:
    """What one shard adds to the held-out round: each held-out text's games,
    every card the shard's games showed, and how many records it holds."""

    games: dict[str, set[str]]
    seen: frozenset[str]
    records: int = 0
    #: As :attr:`ShardCoverage.damaged`.
    damaged: str | None = None
    #: As :attr:`ShardCoverage.cut_games`.
    cut_games: int = 0


def count_text_shard(
    shard: Path, *, roots: dict[str, Path], held_out: HeldOutCards,
) -> TextShardCoverage:
    """``held-out text -> the game ids it resolved in``, over one shard.

    Counted as build-corpus counts a held-out text's games (FR-053): from the
    resolution records its refusal rule keeps, through
    :meth:`HeldOutCards.text_acted_through`. A random-seat game counts: it
    goes to the card-disjoint stratum like any game naming a held-out card
    (FR-059). A game its shard holds no game-end marker for is cut and
    counts toward no text, as build-corpus drops it. The cards shown are read
    off the same resolution records, whose snapshots hold every hand,
    graveyard and permanent. Top-level so a
    process pool can pickle it; each process resolves keys through its own
    :func:`~effects.infrastructure.converted_trees.process_sidecars`, because a
    sidecar cache does not pickle.
    """
    from effects.infrastructure.converted_trees import process_sidecars
    from effects.infrastructure.shard_io import SHARD_DAMAGE, iter_shard_matching

    sidecars = process_sidecars(roots)
    card_of = _card_of_script(sidecars)
    games: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    records = 0

    def every_record(head) -> bool:
        """Count every record; parse only the resolutions and the turn
        actions, among which the game-end markers."""
        nonlocal records
        records += 1
        return head.kind in (RecordKind.RESOLUTION.value, RecordKind.TURN_ACTION.value)

    ends = GameEnds()
    damaged = None
    try:
        for record in iter_shard_matching(shard, where=every_record):
            ends.note(record)
            if record.kind is not RecordKind.RESOLUTION:
                continue
            seen.update(fold_card_name(name) for name in cards_shown(record, card_of))
            if record_refusal(record, sidecars.resolution_of) is not None:
                continue
            text = held_out.text_acted_through(acting_keys(record), sidecars.script_text)
            if text is not None:
                games[text].add(record.game_id)
    except SHARD_DAMAGE as exc:
        damaged = str(exc)
    cut = ends.cut
    kept = {text: ids - cut for text, ids in games.items() if ids - cut}
    return TextShardCoverage(kept, frozenset(seen), records, damaged, len(cut))


class TextCoverageCounter(_ShardCounter):
    """Each held-out text's games, each shard counted once across the run.

    A game credited to a held-out text that is no unit of the round, such as
    one only a token script prints, counts toward none: build-corpus credits
    it to that text too, not to a later acting key's.
    """

    def __init__(
        self, roots: dict[str, Path], held_out: HeldOutCards,
        progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__(progress_every)
        self._roots = dict(roots)
        self._held_out = held_out

    def update(
        self, coverage: Mapping[str, TextCoverage], records_dir: Path,
        *, workers: int | None = None,
    ) -> int:
        """Add the games of shards not yet read. Returns how many it read."""
        self.seen = set()

        def add(part: TextShardCoverage) -> None:
            self.seen |= part.seen
            for text, games in part.games.items():
                if text in coverage:
                    coverage[text].games |= games

        count = partial(count_text_shard, roots=self._roots, held_out=self._held_out)
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


@dataclass(kw_only=True)
class TextRounds(CoverageRounds):
    """The held-out round's rounds: each held-out text a unit, awaited only
    when some castable card carries it."""

    coverage: dict[str, TextCoverage]
    #: ``held-out text -> the deckable listed cards carrying it``.
    carriers: Mapping[str, frozenset[str]]
    counter: TextCoverageCounter
    no_decks: ClassVar[str] = "No unfinished held-out text has a card with converted text."

    @cached_property
    def awaited(self) -> dict[str, TextCoverage]:
        # Fixed for the run: castability comes from the consult, once.
        return awaited_texts(self.coverage)

    def weights(self) -> dict[str, float]:
        return rank_by_consult(
            text_deck_weights(self.coverage, self.carriers, self.target),
            self.verdicts,
        )

    def units_decked(self, cards: set[str]) -> set[str]:
        return {
            text for text, carrying in self.carriers.items() if carrying & cards
        }

    def report(self) -> str:
        return text_residues(self.coverage, self.target).render(self.target)


def _held_out_carriers(
    listed: Iterable[str], texts_by_card: Mapping[str, list[str]],
    non_deck: set[str], held_out: HeldOutCards,
) -> dict[str, frozenset[str]]:
    """``held-out text -> its deckable listed carriers``, for each text some
    deckable card carries.

    Held-out texts are decided over every listed card, non-deck ones
    included, or those would read as unlisted carriers and take their texts
    out of the round; only then do non-deck cards leave as carriers. A text
    only listed cards carry is a unit only when ``held_out`` holds it: a held-out
    card's other lines are carried by it alone too, and under ``template`` their
    template is often printed elsewhere, so no game could ever be credited to
    them (:meth:`HeldOutCards.text_acted_through` credits held-out texts only)
    and the round would deck them until each retired.
    """
    carriers = held_out_texts_of(frozenset(listed), texts_by_card)
    return {
        text: cards - non_deck
        for text, cards in carriers.items()
        if cards - non_deck and held_out.holds_text(text)
    }


def _text_coverage(
    carriers: Mapping[str, frozenset[str]], verdicts: Mapping[str, ConsultVerdict],
) -> dict[str, TextCoverage]:
    """One unit per held-out text, castable while any carrier is not judged
    uncastable."""
    return {
        text: TextCoverage(text=text, castable=any(
            verdicts.get(card, ConsultVerdict.UNKNOWN) != ConsultVerdict.UNCASTABLE
            for card in cards
        ))
        for text, cards in carriers.items()
    }


def run_text_coverage(config: CollectCoverageConfig) -> int:
    """The held-out coverage round: listed cards only, keyed by held-out text."""


    trees = config.source_trees.converted()
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", no_converted_cards(trees))
        return 1
    only_cards = config.deck_holdout.only_cards
    listed = read_name_list(only_cards)
    listed_files = {
        name: script for name, script in trees.card_files.items()
        if fold_card_name(name) in listed
    }
    texts = {name: converted[name].text for name in listed_files}
    non_deck = non_deck_names(texts)
    present = frozenset(name for name in listed_files if name not in non_deck)
    held_out = trees.holdout(config.holdout)
    carriers = _held_out_carriers(listed_files, trees.card_texts(), non_deck, held_out)
    if non_deck:
        logger.info(
            "Held-out round: %d listed card(s) of a non-deck type left out",
            len(non_deck),
        )
    if not carriers:
        logger.error(
            "None of the %d card(s) in %s carries a text only listed cards "
            "carry; nothing to cover.", len(listed), only_cards,
        )
        return 1

    verdicts = consult_castability(sorted(present))
    unresolved = unresolvable(present, verdicts)
    missing = frozenset(unresolved.cards)
    present = present - missing
    carriers = {
        text: cards - missing for text, cards in carriers.items() if cards - missing
    }
    coverage = _text_coverage(carriers, verdicts)
    floor = config.min_text_games
    records_dir = config.effect_records
    counter = TextCoverageCounter(config.source_trees.sidecar_roots(), held_out)
    _count_existing(counter, coverage, config)
    logger.info(
        "Held-out round: %d text(s) on %d card(s); %d already at %d games; %d "
        "with no castable carrier",
        len(coverage), len(present),
        sum(1 for unit in coverage.values() if unit.satisfied(floor)), floor,
        sum(1 for unit in coverage.values() if not unit.castable),
    )

    rounds = TextRounds(
        coverage=coverage, carriers=carriers, verdicts=verdicts, unresolved=unresolved,
        counter=counter, records_dir=records_dir, target=floor, workers=config.workers,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items() if name in present
        },
    )
    # The run's own caps, the three decision-sampling rates included: every
    # stratum has one record composition (FR-059a).
    return _play_to_completion(rounds, config, config.caps)
