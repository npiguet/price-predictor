"""``rarity_bucket`` at every edge (T012)."""

from __future__ import annotations

import pytest

from effects.domain.rarity import RARITY_BUCKETS, rarity_bucket


@pytest.mark.parametrize(
    ("games", "bucket"),
    [
        (0, "1"), (1, "1"),
        (2, "2-4"), (4, "2-4"),
        (5, "5-19"), (19, "5-19"),
        (20, "20+"), (10_000, "20+"),
    ],
)
def test_every_edge_lands_in_its_bucket(games, bucket):
    assert rarity_bucket(games) == bucket


def test_the_buckets_are_in_ascending_order():
    assert RARITY_BUCKETS == ("1", "2-4", "5-19", "20+")


def test_a_fractional_effective_count_rounds_down_into_its_bucket():
    assert rarity_bucket(4.9) == "2-4"
