"""The schedule a training run follows: its learning rate, its curriculum, when
it stops, and which shards each epoch reads with how many steps on each.

Every function here is a pure function of the run's flags, its seed and where
the run has got to; none reads the model or the corpus.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from effects.domain.effect_head_layout import FieldSpec, active_fields

# Hardcoded, not flags (FR-095).
LEARNING_RATE = 1e-4
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01


def warmup_steps(*, epochs: int, steps_per_epoch: int) -> int:
    """Linear warmup over the first 5% of the scheduled steps."""
    return max(1, int(epochs * steps_per_epoch * WARMUP_FRACTION))


def learning_rate_at(step: int, *, warmup: int, scale: float = 1.0) -> float:
    """Linear warmup, then constant, times ``--lr-scale``'s ``scale``: early
    stopping ends the run, and a decay would make patience measure the
    schedule instead of the model."""
    if step >= warmup:
        rate = LEARNING_RATE
    else:
        rate = LEARNING_RATE * (step + 1) / warmup
    return rate * scale


@dataclass(frozen=True)
class Curriculum:
    """The sparse field group's schedule (FR-082): off until the first step
    of ``epoch``, on from there.

    An epoch rather than a step, so the field set never changes inside an
    epoch whose end validates the new fields' untrained heads.
    """

    #: ``--curriculum-epoch``; at 1 or below, the group is on from the run's
    #: first step.
    epoch: int

    def fields_for(
        self, epoch: int, present: Collection[str],
    ) -> tuple[FieldSpec, ...]:
        """The field set every step of ``epoch`` trains with, and validation
        scores, given the sampling classes ``present``."""
        return active_fields(present, sparse=epoch >= self.epoch)

    def resets_at(self, epoch: int) -> bool:
        """Whether the stopper forgets its best control score at the start of
        ``epoch``: where the objective gains the sparse fields. Never at a
        first-epoch curriculum, which leaves no earlier best measuring a
        different objective."""
        return self.epoch > 1 and epoch == self.epoch


#: What a stopper's state measures, written into ``resume.pt``: a resume file
#: whose stopper measured something else (the card-disjoint loss it once kept,
#: lower being better) is refused rather than read the wrong way round.
CONTROL_SCORE_MEASURE = "control_score"


class StopperStateError(ValueError):
    """A stopper state written for another measure than the control score."""


class ControlScoreStopper:
    """Keeps the best control score (FR-069): higher is better, and the run
    stops after ``patience`` epochs without a new best."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.since_best = 0

    def update(self, control_score: float) -> bool:
        """Record an epoch's control score; return True when it set a new best.
        A NaN score sets none and counts toward patience."""
        if control_score > self.best:
            self.best = control_score
            self.since_best = 0
            return True
        self.since_best += 1
        return False

    @property
    def best_score(self) -> float | None:
        """The best control score so far; None before any epoch set one."""
        return None if self.best == float("-inf") else self.best

    def state_dict(self) -> dict:
        return {"measure": CONTROL_SCORE_MEASURE, "best": self.best,
                "since_best": self.since_best}

    def load_state_dict(self, state: dict) -> None:
        if state.get("measure") != CONTROL_SCORE_MEASURE:
            raise StopperStateError(
                "the resume file's stopper kept the card-disjoint validation loss, "
                "not the control score; a run selected by its loss cannot continue "
                "under selection by the control score (FR-069)")
        self.best = state["best"]
        self.since_best = state["since_best"]

    def reset(self) -> None:
        """Forget the best: the objective changed, so no earlier score compares."""
        self.best = float("-inf")
        self.since_best = 0

    @property
    def should_stop(self) -> bool:
        """Never with ``patience`` 0: that run trains every epoch it is given."""
        return self.patience > 0 and self.since_best >= self.patience


def epoch_shards(
    shards: Sequence[Path], *, epoch: int, per_epoch: int, seed: int,
) -> list[Path]:
    """The shards epoch ``epoch`` (1-based) reads, drawn across the corpus.

    Drawn at random, not as a contiguous block: the list is in path order, so
    a block is one collection run's correlated games and never reaches the
    families that sort last. Seeded by ``(seed, epoch)`` alone rather than the
    training RNG, so an epoch's shards do not depend on what earlier epochs
    drew.
    """
    if not shards or per_epoch <= 0:
        return []
    # A string, because 3.14's Random rejects a tuple seed.
    draw = random.Random(f"{seed}:{epoch}")
    if per_epoch >= len(shards):
        picks = list(shards)
        draw.shuffle(picks)
        return picks
    return draw.sample(list(shards), per_epoch)


def steps_per_shard(total: int, count: int) -> list[int]:
    """``total`` training steps (batches, not optimizer steps) divided as
    evenly as possible over ``count`` shards."""
    if count <= 0:
        return []
    base, remainder = divmod(max(total, 0), count)
    return [base + (1 if index < remainder else 0) for index in range(count)]
