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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from effects.application.collector_common import (
    CollectorConfig,
    positive_int,
    read_name_list,
)
from effects.application.coverage_counting import CoverageCounter
from effects.application.coverage_rounds import (
    CoverageRounds,
    Unresolved,
    consult_castability,
    count_existing,
    no_converted_cards,
    non_deck_names,
    play_to_completion,
    unresolvable,
)
from effects.application.held_out_round import run_text_coverage
from effects.domain.coverage import (
    HELD_OUT_GAME_FLOOR,
    CardCoverage,
    ConsultVerdict,
    DeckCandidate,
    deck_weights,
    rank_by_consult,
)

if TYPE_CHECKING:
    from effects.infrastructure.converted_trees import ConvertedTrees

logger = logging.getLogger(__name__)

DEFAULT_TARGET_RECORDS = 50
DEFAULT_NO_PROGRESS_ROUNDS = 3

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
    overlap = _first_overlap(directories)
    if overlap is not None:
        index, other_index = overlap
        first = "--effect-records" if other_index == 0 else "--count-from"
        return (
            f"--count-from {directories[index]} overlaps {first} "
            f"{directories[other_index]}: their shards would count twice"
        )
    return next((
        f"--count-from {directory}: no such directory"
        for directory in count_from if not Path(directory).is_dir()
    ), None)


def _first_overlap(directories: list[Path]) -> tuple[int, int] | None:
    """The first ``(index, earlier index)`` of two directories one of which
    holds the other (or is it), resolved; None when none does."""
    resolved = [directory.resolve() for directory in directories]
    return next((
        (index, other_index)
        for index, directory in enumerate(resolved)
        for other_index, other in enumerate(resolved[:index])
        if _overlaps(directory, other)
    ), None)


def _overlaps(directory: Path, other: Path) -> bool:
    return directory == other or directory.is_relative_to(other) or other.is_relative_to(directory)


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
    rounds = _card_rounds(config, config.source_trees.converted())
    if rounds is None:
        return 1
    return play_to_completion(rounds, config)


def _deck_texts(config: CollectCoverageConfig, trees: ConvertedTrees) -> dict[str, str] | None:
    """The converted text of every card the run decks from, the
    ``--wording-cards`` list's alone when given; None (logged) when none has
    converted text."""
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", no_converted_cards(trees))
        return None
    if config.wording_cards is not None:
        converted = wording_cards_of(converted, config.wording_cards)
        if not converted:
            logger.error(
                "None of the cards in %s has converted text under %s",
                config.wording_cards, trees.cards_folder,
            )
            return None
    return {name: card.text for name, card in converted.items()}


def _card_units(
    config: CollectCoverageConfig, trees: ConvertedTrees, texts: Mapping[str, str],
) -> dict[str, CardCoverage] | None:
    """Every deckable card of ``texts`` a unit, the holdout and non-deck types
    left out; None (logged) when none is left."""
    held_out = config.deck_holdout.exclusions()
    non_deck = non_deck_names(texts)
    coverage = build_coverage_units(
        (name for name in texts if name not in non_deck), held_out,
    )
    if not coverage:
        logger.error("No deckable cards under %s", trees.cards_folder)
        return None
    logger.info(
        "Coverage unit: %d cards (%d held out, %d of a non-deck type), target "
        "%d records each",
        len(coverage), len(held_out), len(non_deck), config.target_records,
    )
    return coverage


def _consulted(
    coverage: dict[str, CardCoverage], verdicts: Mapping[str, ConsultVerdict],
) -> Unresolved:
    """Take the cards the consult left unanswered out of ``coverage`` and give
    the rest their verdicts; the cards taken out."""
    unresolved = unresolvable(coverage, verdicts)
    for name in unresolved.cards:
        del coverage[name]
    for name, verdict in verdicts.items():
        if name in coverage:
            coverage[name].verdict = verdict
    return unresolved


def _card_rounds(config: CollectCoverageConfig, trees: ConvertedTrees) -> CardRounds | None:
    """The card run's units, counted as the corpus stands and consulted; None
    (logged) when there is no card to deck."""
    texts = _deck_texts(config, trees)
    coverage = None if texts is None else _card_units(config, trees, texts)
    if coverage is None:
        return None
    counter = CoverageCounter(config.source_trees.sidecar_roots())
    seeded = count_existing(counter, coverage, config)
    logger.info(
        "Counted %d existing shard(s); %d of %d cards already at target",
        seeded,
        sum(1 for c in coverage.values() if c.satisfied(config.target_records)),
        len(coverage),
    )
    verdicts = consult_castability(sorted(coverage))
    unresolved = _consulted(coverage, verdicts)
    if not coverage:
        logger.error("No card under %s is in Forge's card database", trees.cards_folder)
        return None
    return CardRounds(
        coverage=coverage, verdicts=verdicts, counter=counter, unresolved=unresolved,
        records_dir=config.effect_records, target=config.target_records,
        workers=config.workers,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items()
        },
    )


def wording_cards_of[C](converted: Mapping[str, C], path: Path) -> dict[str, C]:
    """The converted cards ``path`` lists, matched by name (folded when loaded); the listed
    names with no converted card are logged, since a typo would otherwise
    shrink the round without a word."""
    listed = read_name_list(path)
    kept = {
        name: card for name, card in converted.items() if name in listed
    }
    missing = listed - kept.keys()
    logger.info(
        "Wording round: %d of %d listed card(s) have converted text", len(kept), len(listed),
    )
    if missing:
        logger.warning(
            "Listed in %s with no converted card: %s", path, ", ".join(sorted(missing)),
        )
    return kept
