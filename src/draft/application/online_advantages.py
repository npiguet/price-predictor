"""The round's learner rewards standardised into the shared, detached advantage
of each seat (data-model §4, FR-009/FR-023).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draft.application.online_round_loader import RoundBatch

# Degenerate-round guard (FR-023): below this reward std the standardisation
# would amplify float noise into arbitrary advantages.
_ADV_STD_EPS = 1e-8
# Advantage-spread reporting bands (FR-014).
_ADV_NEAR_ZERO = 0.1
_ADV_LARGE = 0.5


# --------------------------------------------------------------------------- #
# Reward → advantage (data-model §4, FR-009/FR-023)
# --------------------------------------------------------------------------- #

@dataclass
class AdvantageStats:
    """The round's reward-signal axis (FR-014)."""

    reward_mean: float = 0.0
    reward_std: float = 0.0
    adv_std: float = 0.0
    adv_near_zero_frac: float = 0.0
    adv_large_frac: float = 0.0
    adv_absmax: float = 0.0
    degenerate: bool = False
    reason: str | None = None


def standardize_round_advantages(
    rewards: list[float],
) -> tuple[list[float], AdvantageStats]:
    """Centre and scale the round's learner rewards to unit variance.

    Returns ``(advantages, stats)`` in reward order. A **degenerate** round
    (fewer than two surviving rewards, or a reward std below ``1e-8``) returns
    no advantages and ``stats.degenerate``: the caller takes no optimizer step,
    so the weights do not move (FR-023). This is the guard that keeps a
    zero-variance round from dividing by zero and amplifying float noise into
    arbitrary advantages.
    """
    n = len(rewards)
    if n < 2:
        return [], AdvantageStats(
            reward_mean=float(np.mean(rewards)) if n else 0.0,
            degenerate=True,
            reason=f"{n} surviving learner reward{'' if n == 1 else 's'}",
        )
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < _ADV_STD_EPS:
        return [], AdvantageStats(
            reward_mean=mean, reward_std=std, degenerate=True,
            reason="zero-variance learner rewards",
        )
    adv = (arr - mean) / std
    abs_adv = np.abs(adv)
    return [float(a) for a in adv], AdvantageStats(
        reward_mean=mean,
        reward_std=std,
        adv_std=float(adv.std()),
        adv_near_zero_frac=float((abs_adv < _ADV_NEAR_ZERO).mean()),
        adv_large_frac=float((abs_adv > _ADV_LARGE).mean()),
        adv_absmax=float(abs_adv.max()),
    )


def assign_advantages(batch: RoundBatch) -> AdvantageStats:
    """Standardise the round's rewards and stamp each seat's shared ``A``.

    The reward is terminal (γ=1), so one detached scalar covers all of a seat's
    picks. On a degenerate round every advantage stays ``0.0`` and the caller
    skips the update entirely.
    """
    advantages, stats = standardize_round_advantages(batch.learner_rewards)
    if stats.degenerate:
        return stats
    for seat_examples, advantage in zip(batch.seat_examples, advantages):
        for example in seat_examples:
            example.advantage = advantage
    return stats
