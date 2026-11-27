"""Gate 2's moves: how much of the right move each keyword's prediction makes.

Each scored effect's predicted move ``m`` (present minus stripped, the
quantity whose sign gate 2's agreement reads) is set beside the right move
``t`` the record's own events give (:func:`effects.domain.keyword_moves.right_move`).
A combat scores the share ``m / t``; an effect read at several subjects
averages their shares first, so a combat is one observation. A combat whose
right move is 0 at every subject has no share and is counted apart, with the
mean ``|m|`` the prediction moved anyway. The summary is reported beside the
verdict and routes nothing (decision D11).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import NamedTuple

import numpy as np

from effects.domain.damage_step_keywords import (
    CombatParticipant,
    DamageStepKeyword,
    KeywordEffect,
)
from effects.domain.evaluation_report import EffectMoves, KeywordMoves
from effects.domain.keyword_moves import right_move
from effects.domain.score_scales import clustered_mean_standard_error

_QUARTILES = (0.25, 0.5, 0.75)


class PredictedMove(NamedTuple):
    """One effect's predicted move at one subject, keyword present minus stripped."""

    effect: KeywordEffect
    subject: str
    predicted: float


@dataclass(slots=True)
class EffectMoveTally:
    """One effect's combats as the pass reads them: each sized combat's share
    and game, and each zero-size combat's mean ``|m|``."""

    shares: list[float] = field(default_factory=list)
    games: list[str] = field(default_factory=list)
    zero_size_moves: list[float] = field(default_factory=list)

    def fold(self, game: str, pairs: Sequence[tuple[float, float]]) -> None:
        """Fold in one combat's ``(m, t)`` at each subject the effect is read at."""
        sized = [predicted / right for predicted, right in pairs if right != 0]
        if sized:
            self.shares.append(fmean(sized))
            self.games.append(game)
        else:
            self.zero_size_moves.append(fmean(abs(predicted) for predicted, _ in pairs))

    def summary(self, label: str) -> EffectMoves:
        zero = self.zero_size_moves
        zero_move = fmean(zero) if zero else float("nan")
        if not self.shares:
            return EffectMoves(label, 0, float("nan"), float("nan"),
                               (float("nan"),) * 3, len(zero), zero_move)
        q1, q2, q3 = (float(q) for q in np.quantile(self.shares, _QUARTILES))
        return EffectMoves(
            label, len(self.shares), fmean(self.shares),
            clustered_mean_standard_error(self.shares, self.games),
            (q1, q2, q3), len(zero), zero_move,
        )


@dataclass(slots=True)
class MoveTally:
    """One keyword's moves, tallied per effect label while the pass reads."""

    per_effect: dict[str, EffectMoveTally] = field(default_factory=dict)

    def fold(self, record, participant: CombatParticipant,
             moves: Iterable[PredictedMove]) -> None:
        """Fold in one observation of ``record``: each move beside its right
        move, read from the record's events."""
        events = tuple(record.payload.events)
        by_label: dict[str, list[tuple[float, float]]] = {}
        for move in moves:
            right = right_move(move.effect, participant, move.subject, events)
            by_label.setdefault(move.effect.label, []).append((move.predicted, right))
        self.add(record.game_id, by_label)

    def add(self, game: str, by_label: dict[str, list[tuple[float, float]]]) -> None:
        """Fold in one combat of ``game``: each effect label's ``(m, t)`` at
        every subject it is read at, ``t`` whatever right move the caller sets
        beside the prediction."""
        for label, pairs in by_label.items():
            self.per_effect.setdefault(label, EffectMoveTally()).fold(game, pairs)

    def summary(self, row: DamageStepKeyword) -> KeywordMoves:
        """The keyword's report, its effects in the row's order."""
        labels = dict.fromkeys(effect.label for effect in row.effects)
        return KeywordMoves(tuple(
            self.per_effect[label].summary(label) for label in labels
            if label in self.per_effect
        ))
