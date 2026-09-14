from __future__ import annotations

import pytest

from effects.domain.corpus_curation import (
    CapHeap, class_targets, keeps, record_hash,
)


def test_record_hash_is_stable_and_seed_dependent():
    assert record_hash("run.0-a.7", seed=42) == record_hash("run.0-a.7", seed=42)
    assert record_hash("run.0-a.7", seed=42) != record_hash("run.0-a.7", seed=43)


def test_record_hash_fits_in_64_bits():
    assert 0 <= record_hash("run.0-a.7", seed=42) < 2**64


def test_a_heap_under_its_cap_sets_no_threshold():
    heap = CapHeap(3)
    for value in (10, 20):
        heap.offer(value)
    assert heap.threshold() is None


def test_a_full_heap_keeps_the_cap_smallest_values():
    heap = CapHeap(3)
    for value in (50, 10, 40, 20, 30):
        heap.offer(value)
    assert sorted(heap.values()) == [10, 20, 30]
    assert heap.threshold() == 30


def test_merging_two_heaps_keeps_the_cap_smallest_of_the_union():
    left, right = CapHeap(3), CapHeap(3)
    for value in (50, 10, 40):
        left.offer(value)
    for value in (5, 60, 35):
        right.offer(value)
    left.merge(right)
    assert sorted(left.values()) == [5, 10, 35]


def test_keeps_admits_exactly_the_values_at_or_below_the_threshold():
    assert keeps(30, 30) is True
    assert keeps(31, 30) is False
    assert keeps(10**19, None) is True


def test_class_targets_are_set_by_the_scarcest_class():
    # rewrite supplies 100 against a 10% share, so the whole dataset is 1000.
    targets, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
    )
    assert targets == {"rewrite": 100, "combat": 900}
    assert shortfall == {}


def test_class_targets_never_ask_for_more_than_a_class_holds():
    targets, _ = class_targets(
        {"rewrite": 10, "combat": 10}, {"rewrite": 0.5, "combat": 0.5},
    )
    assert all(targets[name] <= 10 for name in targets)


def test_a_ceiling_scales_every_class_down_together():
    targets, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
        ceiling=500,
    )
    assert targets == {"rewrite": 50, "combat": 450}
    assert shortfall == {}


def test_a_ceiling_above_what_the_mixture_supports_is_reported_as_shortfall():
    _, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
        ceiling=4000,
    )
    assert shortfall == {"rewrite": 300, "combat": 2700}


def test_a_class_the_corpus_lacks_entirely_drops_out_of_the_mixture():
    targets, _ = class_targets({"combat": 500}, {"rewrite": 0.1, "combat": 0.9})
    assert "rewrite" not in targets
    assert targets["combat"] == 500


def test_class_targets_rejects_a_mixture_that_sums_to_nothing():
    with pytest.raises(ValueError, match="no positive share"):
        class_targets({"combat": 5}, {"combat": 0.0})
