from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.corpus_curation import (
    CapHeap,
    CopyPlan,
    class_targets,
    copies_for_text,
    copy_plan,
    game_disjoint_draw,
    in_game_disjoint_stratum,
    outcome_signature,
    rate_plan,
    record_hash,
    signature_label,
    text_capacity,
)


def test_record_hash_is_stable_and_seed_dependent():
    assert record_hash("run.0-a.7", seed=42) == record_hash("run.0-a.7", seed=42)
    assert record_hash("run.0-a.7", seed=42) != record_hash("run.0-a.7", seed=43)


def test_record_hash_fits_in_64_bits():
    assert 0 <= record_hash("run.0-a.7", seed=42) < 2**64


def test_a_heap_under_its_cap_keeps_everything():
    heap = CapHeap(3)
    for value in (10, 20):
        heap.offer(value)
    assert sorted(heap.values()) == [10, 20]


def test_a_full_heap_keeps_the_cap_smallest_values():
    heap = CapHeap(3)
    for value in (50, 10, 40, 20, 30):
        heap.offer(value)
    assert sorted(heap.values()) == [10, 20, 30]


def test_merging_two_heaps_keeps_the_cap_smallest_of_the_union():
    left, right = CapHeap(3), CapHeap(3)
    for value in (50, 10, 40):
        left.offer(value)
    for value in (5, 60, 35):
        right.offer(value)
    left.merge(right)
    assert sorted(left.values()) == [5, 10, 35]


def test_merging_heaps_with_mismatched_caps_raises_error():
    left, right = CapHeap(5), CapHeap(2)
    for value in (100, 101, 102, 103, 104):
        left.offer(value)
    for value in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        right.offer(value)
    with pytest.raises(ValueError, match="cap"):
        left.merge(right)


def test_merging_equal_cap_heaps_with_overlapping_ranges():
    left, right = CapHeap(3), CapHeap(3)
    for value in (5, 10, 15):
        left.offer(value)
    for value in (2, 8, 12):
        right.offer(value)
    left.merge(right)
    assert sorted(left.values()) == [2, 5, 8]


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


# ── gen-2: copy counts, signatures, stratum placement (T074–T076) ────────


class TestCopiesForText:
    """Spec Story 4 scenario 4: 10, 150 and 1,000 records at cap 200, reuse 4."""

    def test_a_ten_record_text_is_repeated_four_times(self):
        copies = copies_for_text(range(10), 200, text_cap=200, reuse_cap=4)
        assert set(copies.values()) == {4}
        assert sum(copies.values()) == 40

    def test_a_150_record_text_spreads_200_copies_evenly(self):
        copies = copies_for_text(range(150), 200, text_cap=200, reuse_cap=4)
        assert sum(copies.values()) == 200
        assert set(copies.values()) == {1, 2}
        # The extra copies land on the smallest hashes.
        assert all(copies[v] == 2 for v in range(50))

    def test_a_text_over_the_cap_gives_its_cap_smallest_once_each(self):
        copies = copies_for_text(range(1000), 200, text_cap=200, reuse_cap=4)
        assert sum(copies.values()) == 200
        assert {v for v, n in copies.items() if n} == set(range(200))
        assert max(copies.values()) == 1

    def test_a_quota_below_the_distinct_count_takes_the_smallest_hashes(self):
        copies = copies_for_text([30, 10, 20, 40], 2, text_cap=200, reuse_cap=4)
        assert copies == {10: 1, 20: 1, 30: 0, 40: 0}

    def test_the_quota_is_bounded_by_capacity(self):
        assert sum(copies_for_text(range(3), 999, text_cap=200, reuse_cap=4).values()) == 12


def test_text_capacity_is_reuse_times_distinct_under_the_cap():
    assert text_capacity(10, text_cap=200, reuse_cap=4) == 40
    assert text_capacity(150, text_cap=200, reuse_cap=4) == 200
    assert text_capacity(150, text_cap=0, reuse_cap=4) == 600


def test_a_copy_plan_needs_only_a_base_and_a_threshold():
    plan = copy_plan([5, 1, 9], 3, 7)
    assert plan == CopyPlan(base=2, threshold=1)
    assert [plan.copies(v) for v in (1, 5, 9)] == [3, 2, 2]
    assert copy_plan([], 0, 5) == CopyPlan()


def test_a_rate_plan_writes_its_quota_in_expectation():
    plan = rate_plan(10_000, 2_500)
    total = sum(plan.copies(record_hash(f"r{i}", seed=1)) for i in range(10_000))
    assert 2_300 < total < 2_700
    assert rate_plan(100, 300) == CopyPlan(base=3)


_FIXTURE = Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-records.jsonl.gz"


class TestOutcomeSignature:
    def _effect(self):
        from effects.infrastructure.record_io import read_shard

        return next(
            r for r in read_shard(_FIXTURE)
            if r.moment is not None and r.moment.value == "resolution" and r.payload.events
        )

    def test_a_wipe_of_three_and_of_seven_share_a_signature(self):
        """A set over the affected entities, not a multiset (FR-049 step 2)."""
        import dataclasses

        from effects.domain.event_schema import Event, EventType

        record = self._effect()

        def wipe(count):
            events = tuple(
                Event(type=EventType.DESTROYED, subjects=(f"E{i}",)) for i in range(count)
            )
            return dataclasses.replace(record, payload=dataclasses.replace(
                record.payload, events=events,
            ))

        assert outcome_signature(wipe(3)) == outcome_signature(wipe(7)) == frozenset({
            ("died", True),
        })

    def test_a_real_effect_halfs_signature_is_stable(self):
        record = self._effect()
        assert outcome_signature(record) == outcome_signature(record)
        assert signature_label(outcome_signature(record))

    def test_the_label_is_sorted_and_readable(self):
        assert signature_label(frozenset({("stayed", True), ("died", True)})) == (
            "died:changed,stayed:changed"
        )
        assert signature_label(frozenset()) == "(empty)"


class TestGameDisjointPlacement:
    """Spec Story 4 scenarios 5 and 6."""

    def test_a_game_below_the_share_enters(self):
        game = next(f"g{i}" for i in range(1000) if game_disjoint_draw(f"g{i}") < 0.01)
        assert in_game_disjoint_stratum(game, False, False, 0.01, 0.15)

    def test_a_keyword_combat_game_uses_the_keyword_share(self):
        game = next(
            f"g{i}" for i in range(1000) if 0.01 <= game_disjoint_draw(f"g{i}") < 0.15
        )
        assert not in_game_disjoint_stratum(game, False, False, 0.01, 0.15)
        assert in_game_disjoint_stratum(game, False, True, 0.01, 0.15)

    def test_a_game_naming_a_held_out_card_never_enters(self):
        assert not in_game_disjoint_stratum("g0", True, True, 1.0, 1.0)

    def test_placement_depends_on_the_game_alone(self):
        games = [f"g{i}" for i in range(2000)]
        first = {g for g in games[:500] if in_game_disjoint_stratum(g, False, False, 0.05, 0.15)}
        grown = {g for g in games if in_game_disjoint_stratum(g, False, False, 0.05, 0.15)}
        assert first <= grown

    def test_the_draw_is_crc32_of_the_game_id(self):
        import zlib

        assert game_disjoint_draw("run.0-a.5") == zlib.crc32(b"run.0-a.5") % 10**6 / 10**6
