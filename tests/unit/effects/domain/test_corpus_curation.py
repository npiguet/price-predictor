from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.corpus_curation import (
    CapHeap,
    CopyPlan,
    KeyedCapHeap,
    class_targets,
    copy_plan,
    game_disjoint_draw,
    game_hash,
    outcome_signature,
    rate_plan,
    record_hash,
    signature_label,
    text_capacity,
)


def copies_for_text(record_hashes, quota, text_cap, reuse_cap) -> dict[int, int]:
    """``record hash -> copies`` for one text in one cell: the quota bounded by
    the text's capacity, then placed by the copy plan, as the write pass does."""
    hashes = sorted(set(record_hashes))
    capacity = text_capacity(len(hashes), text_cap=text_cap, reuse_cap=reuse_cap)
    plan = copy_plan(hashes, len(hashes), min(quota, capacity))
    return {value: plan.copies(value) for value in hashes}


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


def test_a_heap_keeps_items_with_their_values_and_breaks_ties_on_the_tiebreak():
    left, right = KeyedCapHeap(3), KeyedCapHeap(3)
    left.offer(30, "c", (0,))
    left.offer(10, "a", (1,))
    right.offer(10, "b", (2,))
    right.offer(20, "z", (0,))
    left.merge(right)
    assert left.items() == ["b", "a", "z"]
    assert len(left) == 3


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


_FIXTURE = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"


class TestOutcomeSignature:
    def _effect(self):
        from effects.infrastructure.shard_io import read_shard

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


class TestGameDisjointDraw:
    def test_the_draw_is_crc32_of_the_game_id(self):
        import zlib

        assert game_disjoint_draw("run.0-a.5") == zlib.crc32(b"run.0-a.5") % 10**6 / 10**6

    def test_game_hash_is_stable_across_calls(self):
        """Pinned to the algorithm, not merely to idempotency (FR-088a).

        ``game_hash("x") == game_hash("x")`` is also true of the built-in
        ``hash()``, whose salt is per-process rather than per-call, so that
        assertion alone cannot fail for the defect the function warns against.
        """
        import zlib

        assert game_hash("run.0-a.3") == zlib.crc32(b"run.0-a.3")
        assert game_hash("run.0-a.3") == game_hash("run.0-a.3")


# ── condition balance (FR-056; story 6 acceptance 3) ────────────────────


_KINGS_ASSASSIN = (
    "AB$ Destroy | ActivationFirstCombat$ True | ActivationPhases$ Upkeep->BeginCombat"
    " | Cost$ T | PlayerTurn$ True | ValidTgts$ Creature.tapped"
)
_TURN_CLAUSE = (
    "SP$ Pump | ConditionPresent$ Card.Mount | NumAtt$ +1 | SubAbility$ SV1"
    " [SEG] SV1: DB$ Draw | ConditionPlayerTurn$ True | NumCards$ 1"
)
_TURN_STATIC = "Mode$ Continuous | Affected$ Creature.YouCtrl | AddPower$ 1 | Condition$ PlayerTurn"


class _Lines:
    """A sidecar lookup over ``key -> script text``."""

    def __init__(self, scripts):
        from effects.domain.provenance import SidecarLine

        self._lines = {
            key: SidecarLine(line_index=0, line_kind="spell", provenance=(key,),
                             script_text=text)
            for key, text in scripts.items()
        }

    def line_for(self, key):
        return self._lines.get(key)

    def first_line(self, keys):
        return next((self._lines[k] for k in keys if k in self._lines), None)


def _key(name, kind="spell"):
    from effects.domain.provenance import ProvenanceKey

    return ProvenanceKey(f"cardsfolder/x/{name}.txt", 0, kind, 0)


def _decision(key, failed):
    from effects.domain.records import (
        Candidate,
        FailedCheck,
        PlayabilityDecisionPayload,
        RecordKind,
    )
    from tests.unit.effects.builders import make_record

    candidate = Candidate(
        ability=(key,), can_play=not failed, affordable=True, has_legal_target=True,
        failed_checks=tuple(FailedCheck(check) for check in failed),
    )
    return make_record(RecordKind.PLAYABILITY, generation=3,
                       payload=PlayabilityDecisionPayload(candidates=(candidate,)))


def _resolution(key, conditions):
    from effects.domain.records import ConditionMet, ResolutionPayload
    from tests.unit.effects.builders import make_record

    return make_record(generation=3, ability=(key,), payload=ResolutionPayload(
        conditions_met=tuple(ConditionMet(segment, met) for segment, met in conditions),
    ))


def _continuous(key, active):
    from effects.domain.records import RecordKind
    from effects.domain.state_snapshot import CardLine
    from tests.unit.effects.builders import make_entity, make_record, make_snapshot

    host = make_entity("E1", "anthem", lines=(
        CardLine(keys=(key,), kind="static", granted=False, active=active),
    ))
    return make_record(RecordKind.CONTINUOUS, generation=3, ability=(key,),
                       state=make_snapshot(entities=(host,)))


class TestConditionBalance:
    def test_a_candidates_turn_restriction_held_unless_a_timing_check_failed(self):
        from effects.domain.corpus_curation import ConditionSide, turn_step_condition

        key = _key("kings_assassin")
        lines = _Lines({key: _KINGS_ASSASSIN})
        assert turn_step_condition(_decision(key, ()), lines).side is ConditionSide.MET
        held = turn_step_condition(_decision(key, ("cost",)), lines)
        assert held.side is ConditionSide.MET and held.carrier == key
        for check in ("whose_turn", "phase", "combat"):
            assert turn_step_condition(_decision(key, (check,)), lines).side is (
                ConditionSide.UNMET)

    def test_a_candidate_with_no_turn_restriction_carries_none(self):
        from effects.domain.corpus_curation import turn_step_condition

        key = _key("shock")
        lines = _Lines({key: "SP$ DealDamage | NumDmg$ 2 | SorcerySpeed$ True"})
        assert turn_step_condition(_decision(key, ("casting_speed",)), lines) is None

    def test_a_resolved_clause_reads_its_own_segments_condition(self):
        from effects.domain.corpus_curation import ConditionSide, turn_step_condition

        key = _key("pump_then_draw")
        lines = _Lines({key: _TURN_CLAUSE})
        # The root's ConditionPresent$ is no turn condition; segment 0 is SV1.
        record = _resolution(key, [("root", True), ("0", False)])
        assert turn_step_condition(record, lines).side is ConditionSide.UNMET
        assert turn_step_condition(_resolution(key, [("root", True)]), lines) is None
        assert turn_step_condition(_resolution(key, []), lines) is None

    def test_a_listed_static_reads_its_lines_active(self):
        from effects.domain.corpus_curation import ConditionSide, turn_step_condition

        key = _key("anthem", "static")
        lines = _Lines({key: _TURN_STATIC})
        assert turn_step_condition(_continuous(key, False), lines).side is ConditionSide.UNMET
        assert turn_step_condition(_continuous(key, True), lines).side is ConditionSide.MET
        assert turn_step_condition(_continuous(key, None), lines) is None

    def test_the_condition_joins_the_outcome_signature(self):
        from effects.domain.corpus_curation import cell_signature, outcome_signature
        from effects.domain.sampling import SamplingClass

        key = _key("kings_assassin")
        lines = _Lines({key: _KINGS_ASSASSIN})
        decision = SamplingClass.PLAYABILITY_DECISION
        assert cell_signature(_decision(key, ()), decision, lines) == "condition:met"
        assert cell_signature(_decision(key, ("phase",)), decision, lines) == "condition:unmet"

        clause = _key("pump_then_draw")
        effect = _resolution(clause, [("0", True)])
        plain = signature_label(outcome_signature(effect))
        assert cell_signature(effect, SamplingClass.RESOLUTION_EFFECT, _Lines(
            {clause: _TURN_CLAUSE})) == f"{plain}|condition:met"
        assert cell_signature(effect, SamplingClass.RESOLUTION_EFFECT, _Lines({})) == plain

    def test_met_and_unmet_draw_equal_budgets_a_short_side_repeating(self):
        from effects.domain.corpus_plan import Cell, Member, plan_selection
        from effects.domain.sampling import SamplingClass

        klass = SamplingClass.PLAYABILITY_DECISION
        met = Cell(klass, family="Destroy", signature="condition:met")
        unmet = Cell(klass, family="Destroy", signature="condition:unmet")
        plan = plan_selection(
            {met: {"": Member(distinct=10)}, unmet: {"": Member(distinct=2)}},
            mix={klass: 1.0}, ceiling=12, text_cap=0, reuse_cap=4,
        )
        assert plan.keyless[unmet].base == 3
        assert plan.keyless[met] == rate_plan(10, 6)

    def test_a_condition_seen_on_one_side_only_is_named(self):
        from effects.domain.corpus_curation import (
            ConditionSide,
            TurnStepCondition,
            condition_key,
            one_sided_conditions,
        )
        from effects.domain.sampling import SamplingClass

        both, one = _key("kings_assassin"), _key("anthem", "static")
        klass = SamplingClass.PLAYABILITY_DECISION
        counts = {
            condition_key(klass, TurnStepCondition(both, ConditionSide.MET)): 4,
            condition_key(klass, TurnStepCondition(both, ConditionSide.UNMET)): 1,
            condition_key(klass, TurnStepCondition(one, ConditionSide.MET)): 7,
        }
        assert one_sided_conditions(counts) == {str(klass): [one.label]}



class TestValidationClassMix:
    """FR-059a (T183): each validation stratum is written in the training
    stratum's class mix. A validation record is never dropped, so the
    over-represented class sets the size and the others are repeated, at most
    ``--reuse-cap`` times a record, as training repeats a short family's."""

    def test_the_multipliers_deliver_the_mix(self):
        from effects.domain.corpus_curation import validation_multipliers

        counts = {"combat": 60, "rewrite": 10, "trigger": 30}
        mix = {"combat": 0.5, "rewrite": 0.25, "trigger": 0.25}

        multipliers = validation_multipliers(counts, mix, reuse_cap=10)

        written = {name: counts[name] * multipliers[name] for name in counts}
        total = sum(written.values())
        assert {name: n / total for name, n in written.items()} == pytest.approx(mix)
        assert min(multipliers.values()) == pytest.approx(1.0)

    def test_no_record_is_dropped_and_none_repeats_past_the_cap(self):
        from effects.domain.corpus_curation import validation_multipliers

        multipliers = validation_multipliers(
            {"combat": 100, "rewrite": 1}, {"combat": 0.5, "rewrite": 0.5}, reuse_cap=4,
        )

        assert multipliers == {"combat": 1.0, "rewrite": 4.0}

    def test_a_class_the_mix_omits_is_written_as_collected(self):
        from effects.domain.corpus_curation import validation_multipliers

        multipliers = validation_multipliers(
            {"combat": 10, "continuous": 5}, {"combat": 1.0}, reuse_cap=4,
        )

        assert multipliers["continuous"] == 1.0

    def test_copies_realize_a_fractional_multiplier_on_average(self):
        from effects.domain.corpus_curation import natural_copies, record_hash

        copies = [natural_copies(2.25, record_hash(f"r{i}", seed=1)) for i in range(4000)]

        assert set(copies) == {2, 3}
        assert sum(copies) / len(copies) == pytest.approx(2.25, abs=0.03)
