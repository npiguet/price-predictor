"""The directed-fork probability of a rule from its two sides' counts."""

from __future__ import annotations

import pytest

from price_predictor.domain.directed_fork_probabilities import (
    MIN_NEEDED_PROBABILITY,
    directed_fork_probability,
)


class TestTheBoundaries:
    def test_both_sides_at_their_count_is_never_forked(self):
        assert directed_fork_probability((100, 100), 100) == 0.0

    def test_both_sides_past_their_count_is_never_forked(self):
        assert directed_fork_probability((250, 100), 100) == 0.0

    def test_a_side_at_zero_always_forks(self):
        assert directed_fork_probability((0, 500), 100) == 1.0
        assert directed_fork_probability((500, 0), 100) == 1.0
        assert directed_fork_probability((0, 0), 100) == 1.0

    def test_in_between_it_is_the_missing_share_of_the_shorter_side(self):
        assert directed_fork_probability((40, 500), 100) == pytest.approx(0.6)
        assert directed_fork_probability((500, 25), 100) == pytest.approx(0.75)

    def test_a_nearly_met_rule_keeps_the_floor_probability(self):
        assert directed_fork_probability((99, 100), 100) == MIN_NEEDED_PROBABILITY
        assert MIN_NEEDED_PROBABILITY == 0.05

    def test_the_shorter_side_decides(self):
        assert directed_fork_probability((10, 90), 100) == directed_fork_probability(
            (90, 10), 100
        )


def test_a_count_below_one_is_refused():
    with pytest.raises(ValueError, match="count"):
        directed_fork_probability((1, 1), 0)
