"""What ``collect-coverage``'s card run and held-out round share: the rounds'
units as the round loop reads them, the castability consult and the cards it
cannot answer, the first count of what the corpus already holds, and the run
under a supervisor with the exit code its rounds earn.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from effects.application.collection_rounds import (
    RoundCharge,
    RoundDecks,
    RoundPlan,
    play_rounds,
)
from effects.application.collector_common import (
    build_coverage_decks,
    collector_supervisor,
    record_damaged_shards,
)
from effects.application.coverage_counting import CoverageCounter, TextCoverageCounter
from effects.domain.coverage import (
    ConsultVerdict,
    CoverageUnit,
    DeckCandidate,
    is_complete,
    is_deckable,
)

if TYPE_CHECKING:
    from effects.application.collect_coverage import CollectCoverageConfig
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.converted_trees import ConvertedTrees

logger = logging.getLogger(__name__)


def no_converted_cards(trees: ConvertedTrees) -> str:
    """The error for a run whose card tree holds no converted card. Deck
    candidates and the coverage unit come from the card tree alone (FR-046,
    FR-048): a token script could never be satisfied or retired."""
    where = trees.cards_folder or ", ".join(str(folder) for folder in trees.roots.values())
    return (
        f"No converted cards under {where}. Run 'python -m price_predictor "
        "convert' first."
    )


def non_deck_names(texts: Mapping[str, str]) -> set[str]:
    """The cards whose converted text names a type no deck holds."""
    return {name for name, text in texts.items() if not is_deckable(text)}


@dataclass(frozen=True)
class Unresolved:
    """The cards Forge's card database does not hold, out of the pool the
    consult was asked about."""

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
    label: ClassVar[str] = "coverage"
    stopping: ClassVar[str] = (
        "Interrupted after round %d; stopping instead of starting another round"
    )

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

    @abstractmethod
    def report(self) -> str:
        """What the rounds could not reach, printed however they ended."""

    def before_rounds(self, supervisor: CollectorSupervisor) -> None:
        """A coverage run tells the supervisor nothing before its rounds."""

    def decks(self, weights: dict[str, float], plan: RoundPlan) -> RoundDecks:
        return RoundDecks(build_coverage_decks(
            weights, self.candidates, plan.decks_per_round, rng=plan.rng,
        ))

    def count_round(self, round_number: int) -> set[str]:
        """Count the shards the last round wrote into ``units``; the cards
        their games showed."""
        read = self.counter.update(self.coverage, self.records_dir, workers=self.workers)
        logger.info("Counted %d new shard(s) from round %d", read, round_number)
        return self.counter.seen

    def after_count(self, supervisor: CollectorSupervisor) -> None:
        """A coverage run tells the supervisor nothing after a round."""

    def charged(self, cards: set[str], shown: set[str]) -> RoundCharge:
        return RoundCharge(self.units_decked(cards & shown), self.units_decked(cards))

    def summary(self, round_number: int, retired: list[str]) -> str:
        satisfied = sum(1 for unit in self.units.values() if unit.satisfied(self.target))
        remaining = sum(1 for unit in self.awaited.values() if not unit.done(self.target))
        return (
            f"round {round_number}: {satisfied} satisfied, {remaining} remaining, "
            f"{len(retired)} retired this round"
        )


def play_to_completion(rounds: CoverageRounds, config: CollectCoverageConfig) -> int:
    """Play ``rounds`` under a supervisor and report what they left; 130 when
    Ctrl-C ended them, 1 when they played and wrote no record at all."""
    with collector_supervisor(config) as supervisor:
        if rounds.unresolved.pool:
            supervisor.record(**rounds.unresolved.fields())
        record_damaged_shards(supervisor, rounds.counter.damaged)
        supervisor.count_cut_games(rounds.counter.cut_games)
        played = play_rounds(rounds, supervisor, RoundPlan(
            config.decks_per_round, config.no_progress_rounds, random.Random(config.seed),
        ))
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


def count_existing(
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
    it until it retired unseen; it leaves the unit and the decks
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
