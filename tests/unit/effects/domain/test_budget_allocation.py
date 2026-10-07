"""``allocate``: equal shares, capped by capacity, redistributed (T073, FR-049)."""

from __future__ import annotations

from effects.domain.budget_allocation import allocate


def test_a_short_member_is_filled_and_its_remainder_shared():
    """Spec Story 4 scenario 1: 4,000 over four, one with capacity 300."""
    shares = allocate(4000, {"a": 10_000, "b": 10_000, "c": 10_000, "short": 300})

    assert shares["short"] == 300
    assert sum(shares.values()) == 4000
    rest = [shares[k] for k in "abc"]
    assert max(rest) - min(rest) <= 1


def test_redistribution_repeats_until_every_member_is_full():
    shares = allocate(100, {"a": 5, "b": 20, "c": 1000})

    assert shares == {"a": 5, "b": 20, "c": 75}


def test_no_members_gives_nothing():
    assert allocate(500, {}) == {}


def test_a_member_with_no_capacity_takes_nothing():
    assert allocate(10, {"a": 0, "b": 10}) == {"a": 0, "b": 10}


def test_a_budget_above_total_capacity_fills_everyone_and_leaves_a_shortfall():
    shares = allocate(1000, {"a": 100, "b": 200})

    assert shares == {"a": 100, "b": 200}
    assert 1000 - sum(shares.values()) == 700


def test_integer_remainders_sum_exactly_to_the_budget():
    for budget in range(0, 50):
        shares = allocate(budget, {"x": 100, "y": 100, "z": 100})
        assert sum(shares.values()) == budget


def test_the_remainder_goes_by_sorted_key_so_the_split_is_deterministic():
    assert allocate(2, {"b": 10, "a": 10, "c": 10}) == {"a": 1, "b": 1, "c": 0}


def test_set_valued_members_are_ordered_independently_of_string_hashing():
    """Outcome signatures are frozensets, whose repr follows the hash seed."""
    import os
    import subprocess
    import sys

    script = (
        "from effects.domain.budget_allocation import allocate;"
        "keys=[frozenset({('died',True),('stayed',False)}), frozenset({('exiled',True)}),"
        " frozenset({('to_hand',True),('blinked',False)})];"
        "print(sorted((sorted(k), v) for k, v in allocate(2, {k: 5 for k in keys}).items()))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "1", "7")
    }
    assert len(runs) == 1
