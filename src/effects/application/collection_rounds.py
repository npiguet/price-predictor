"""The one round loop both collectors play, over a units protocol.

A round draws decks from the units' weights, plays them, counts the shards
it wrote into the units, and retires every unit that stalled. ``collect-
coverage``'s card run and held-out round and ``collect-variants``' rounds
differ only in what a unit is, how its cards are weighted and decked, and
what a round's games showed: :class:`RoundUnits` says those, the loop the
rest.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

from effects.application.collector_common import record_damaged_shards
from effects.domain.coverage import CoverageUnit, is_complete, retire_stalled

if TYPE_CHECKING:
    from pathlib import Path

    from effects.application.shard_ledger import ShardLedger
    from effects.infrastructure.collector_connector import CollectorSupervisor

logger = logging.getLogger(__name__)


class RoundDecks(NamedTuple):
    """A round's decks, and in the steered round the rule each deck is
    aimed at (None otherwise)."""

    decks: list[list[str]]
    rules: list[str | None] | None = None


@dataclass(frozen=True)
class RoundCharge:
    """The units a round's stalls are charged to: those its games showed
    (``decked``), and those its decks held (``written``), which a worker
    drawing its decks with replacement may never have played."""

    decked: set[str]
    written: set[str]


@dataclass(frozen=True)
class RoundPlan:
    """How the rounds are played: ``--decks-per-round`` matches each, a unit
    retiring after ``--no-progress-rounds`` stalled rounds, the decks drawn
    from ``rng``."""

    decks_per_round: int
    no_progress_rounds: int
    rng: random.Random


@dataclass(frozen=True)
class RoundsPlayed:
    """How a run's rounds ended: how many were played, and whether Ctrl-C
    ended them."""

    rounds: int
    interrupted: bool


class RoundUnits(Protocol):
    """What the round loop needs from one kind of unit."""

    @property
    def label(self) -> str:
        """The decks file's kind and the workers' round label."""

    @property
    def no_decks(self) -> str:
        """Logged when the weighted cards yield no deck at all."""

    @property
    def stopping(self) -> str:
        """Logged, with the round's number, when Ctrl-C ended the rounds."""

    @property
    def target(self) -> int: ...

    @property
    def units(self) -> Mapping[str, CoverageUnit]:
        """Every unit, each retiring when it stalls."""

    @property
    def awaited(self) -> Mapping[str, CoverageUnit]:
        """The units whose completion ends the rounds."""

    @property
    def counter(self) -> ShardLedger:
        """The ledger the rounds' shards are counted through."""

    def before_rounds(self, supervisor: CollectorSupervisor) -> None:
        """What the supervisor is told before the first round."""

    def weights(self) -> dict[str, float]:
        """The next round's per-card deck weights."""

    def decks(self, weights: dict[str, float], plan: RoundPlan) -> RoundDecks:
        """The round's decks, drawn from ``weights``."""

    def count_round(self, round_number: int) -> set[str]:
        """Count the shards the round wrote into the units; what its games
        showed."""

    def after_count(self, supervisor: CollectorSupervisor) -> None:
        """What the supervisor is told once a round is counted."""

    def charged(self, cards: set[str], shown: set[str]) -> RoundCharge:
        """The units a round's stalls are charged to, from the cards its
        decks held and what :meth:`count_round` said its games showed."""

    def summary(self, round_number: int, retired: list[str]) -> str:
        """The line a counted round prints."""


def play_rounds(
    rounds: RoundUnits, supervisor: CollectorSupervisor, plan: RoundPlan,
) -> RoundsPlayed:
    """Play rounds until every awaited unit is done, or no deck can be drawn.

    The decks file is the run's own (``decks/<label>-decks.<run_id>.txt``):
    every worker reloads it at its start, so two runs on one records
    directory would otherwise play each other's decks.
    """
    decks_file = supervisor.decks_path(rounds.label)
    rounds.before_rounds(supervisor)
    round_number = 0
    while not is_complete(rounds.awaited, rounds.target):
        previous = {key: unit.progress for key, unit in rounds.units.items()}
        drawn = _draw(rounds, plan)
        if drawn is None:
            break
        round_number += 1
        played = RoundSettling(round_number, previous, drawn, plan)
        _play(supervisor, played, decks_file, rounds.label)
        _settle(rounds, supervisor, played)
        if supervisor.interrupted:
            # ForgeWorkerPool swallows SIGINT, so without this Ctrl-C would only
            # skip to the next round.
            logger.warning(rounds.stopping, round_number)
            return RoundsPlayed(round_number, interrupted=True)
    return RoundsPlayed(round_number, interrupted=False)


def _draw(rounds: RoundUnits, plan: RoundPlan) -> RoundDecks | None:
    """The round's decks, or None, logging why when it is for want of a deck."""
    weights = rounds.weights()
    if not weights:
        return None
    drawn = rounds.decks(weights, plan)
    if not drawn.decks:
        logger.error("%s", rounds.no_decks)
        return None
    return drawn


def _play(
    supervisor: CollectorSupervisor, played: RoundSettling, decks_file: Path, label: str,
) -> None:
    """Play the round's decks; a rule column only for decks aimed at rules."""
    from effects.infrastructure.collector_connector import RoundSpec

    decks, rules = played.drawn
    matches = played.plan.decks_per_round
    supervisor.play_round(decks, decks_file, RoundSpec(label=label, matches=matches, rules=rules))


@dataclass(frozen=True)
class RoundSettling:
    """One played round, as its count reads it: its number, each unit's
    progress before it, its decks, and the plan."""

    round_number: int
    previous: dict[str, int]
    drawn: RoundDecks
    plan: RoundPlan


def _settle(
    rounds: RoundUnits, supervisor: CollectorSupervisor, played: RoundSettling,
) -> None:
    """Count a played round, report what it read, and retire what stalled.
    A stall is charged to what the round's games showed: a worker draws its
    decks with replacement, so some written decks are never played."""
    cards = {name for deck in played.drawn.decks for name in deck}
    shown = rounds.count_round(played.round_number)
    record_damaged_shards(supervisor, rounds.counter.damaged)
    supervisor.count_cut_games(rounds.counter.last_cut)
    rounds.after_count(supervisor)
    charge = rounds.charged(cards, shown)
    retired = retire_stalled(
        rounds.units, played.previous, no_progress_rounds=played.plan.no_progress_rounds,
        decked=charge.decked, written=charge.written,
    )
    logger.info("%s", rounds.summary(played.round_number, retired))
