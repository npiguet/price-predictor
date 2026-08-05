"""Gen-3 reward → advantage: round standardisation + degenerate guards.

Covers data-model §4 and spec FR-008/FR-009/FR-022/FR-023.
"""

from __future__ import annotations

import numpy as np

from draft.application.train_draft_agent_online import (
    learner_seat_rewards,
    standardize_round_advantages,
)
from draft.domain.draft_geometry import Booster, DraftRecord, Seat


def _record(seats: list[Seat], pod: int = 4, pack_size: int = 2) -> DraftRecord:
    boosters = [
        Booster("TST", [f"k{k}c{j}" for j in range(pack_size)]) for k in range(pod)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


# --------------------------------------------------------------------------- #
# Round standardisation
# --------------------------------------------------------------------------- #

def test_standardised_advantages_are_zero_mean_unit_std() -> None:
    rewards = [6.0, 0.0, -2.0, -6.0, 1.5]
    advantages, stats = standardize_round_advantages(rewards)

    assert not stats.degenerate
    assert len(advantages) == len(rewards)
    assert abs(float(np.mean(advantages))) < 1e-6
    assert abs(float(np.std(advantages)) - 1.0) < 1e-6
    assert stats.reward_mean == float(np.mean(rewards))
    assert stats.reward_std == float(np.std(rewards))
    assert abs(stats.adv_std - 1.0) < 1e-6


def test_advantage_ordering_matches_reward_ordering() -> None:
    rewards = [-3.0, 5.0, 1.0]
    advantages, _ = standardize_round_advantages(rewards)
    assert advantages[1] > advantages[2] > advantages[0]


def test_stats_report_advantage_spread_fractions() -> None:
    # Two far-out rewards and three clustered ones: |A| bands are readable.
    rewards = [10.0, -10.0, 0.0, 0.0, 0.0]
    _, stats = standardize_round_advantages(rewards)

    assert stats.adv_near_zero_frac == 3 / 5   # the three |A| < 0.1 entries
    assert stats.adv_large_frac == 2 / 5       # the two |A| > 0.5 entries
    assert stats.adv_absmax > 1.0


# --------------------------------------------------------------------------- #
# Degenerate-round guards (FR-023)
# --------------------------------------------------------------------------- #

def test_fewer_than_two_rewards_is_a_no_op_round() -> None:
    advantages, stats = standardize_round_advantages([4.2])
    assert stats.degenerate
    assert stats.reason is not None
    assert advantages == []


def test_empty_round_is_a_no_op_round() -> None:
    advantages, stats = standardize_round_advantages([])
    assert stats.degenerate
    assert advantages == []


def test_zero_variance_rewards_do_not_divide_by_zero() -> None:
    advantages, stats = standardize_round_advantages([3.0, 3.0, 3.0])
    assert stats.degenerate
    assert advantages == []
    assert stats.reward_std == 0.0


def test_near_zero_variance_is_degenerate_below_the_epsilon() -> None:
    advantages, stats = standardize_round_advantages([1.0, 1.0 + 1e-12])
    assert stats.degenerate
    assert advantages == []


# --------------------------------------------------------------------------- #
# Per-seat reward extraction (FR-008, FR-022)
# --------------------------------------------------------------------------- #

def test_only_learner_seats_contribute_rewards() -> None:
    record = _record([
        Seat("gen-3", ["x"] * 40, 10.0),
        Seat("gen-1", ["x"] * 40, 6.0),
        Seat("gen-3", ["x"] * 40, 2.0),
        Seat("forge-r30", ["x"] * 40, 4.0),
    ])
    rewards, dropped = learner_seat_rewards(record, "gen-3")

    assert [seat for seat, _ in rewards] == [0, 2]
    assert dropped == 0
    # Leave-one-out baseline spans ALL non-failed seats, not just learner ones.
    assert rewards[0][1] == 10.0 - (6.0 + 2.0 + 4.0) / 3
    assert rewards[1][1] == 2.0 - (10.0 + 6.0 + 4.0) / 3


def test_failed_learner_build_is_excluded_and_counted() -> None:
    record = _record([
        Seat("gen-3", ["x"] * 40, 10.0),
        Seat("gen-3", [], None),          # failed build
        Seat("gen-1", ["x"] * 40, 6.0),
        Seat("forge-r30", ["x"] * 40, 2.0),
    ])
    rewards, dropped = learner_seat_rewards(record, "gen-3")

    assert [seat for seat, _ in rewards] == [0]
    assert dropped == 1


def test_learner_seat_with_no_other_scored_seat_is_dropped() -> None:
    # Only one non-failed seat in the whole pod: its leave-one-out baseline is
    # undefined, so it is excluded rather than silently baselined against 0.
    record = _record([
        Seat("gen-3", ["x"] * 40, 10.0),
        Seat("gen-1", [], None),
        Seat("forge-r30", [], None),
        Seat("forge-r100", [], None),
    ])
    rewards, dropped = learner_seat_rewards(record, "gen-3")

    assert rewards == []
    assert dropped == 1


def test_failed_non_learner_seats_do_not_count_as_dropped() -> None:
    record = _record([
        Seat("gen-3", ["x"] * 40, 10.0),
        Seat("gen-1", [], None),          # not a learner seat
        Seat("forge-r30", ["x"] * 40, 2.0),
        Seat("forge-r100", ["x"] * 40, 4.0),
    ])
    rewards, dropped = learner_seat_rewards(record, "gen-3")

    assert len(rewards) == 1
    assert dropped == 0
