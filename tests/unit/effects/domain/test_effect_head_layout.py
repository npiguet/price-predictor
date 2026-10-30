"""The per-entity head's field layout, and which fields a corpus trains."""

from __future__ import annotations

import pytest

from effects.domain.effect_head_input import COUNTER_TYPES
from effects.domain.effect_head_layout import (
    DURATIONS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
    FieldScope,
    FieldSpec,
    active_fields,
)
from effects.domain.sampling import SAMPLING_CLASSES, SamplingClass
from effects.domain.state_snapshot import COLORS


class TestFieldLayout:
    def test_field_slices_tile_the_head_without_gaps_or_overlap(self):
        ordered = sorted(FIELD_SLICES.values())
        assert ordered[0][0] == 1
        for (_, end), (start, _) in zip(ordered, ordered[1:]):
            assert end == start
        assert ordered[-1][1] == PER_ENTITY_WIDTH

    def test_every_field_has_a_slice_of_its_declared_width(self):
        for spec in PER_ENTITY_FIELDS:
            start, end = FIELD_SLICES[spec.name]
            assert end - start == spec.width

    def test_a_categorical_field_is_as_wide_as_its_vocabulary(self):
        assert FIELDS_BY_NAME["zone_outcome"].width == len(ZONE_OUTCOMES)
        assert FIELDS_BY_NAME["pt_duration"].width == len(DURATIONS)

    def test_the_player_group_covers_the_fields_the_spec_names(self):
        player = {s.name for s in PER_ENTITY_FIELDS if s.scope is FieldScope.PLAYER}
        assert {"life_delta", "cards_drawn", "cards_discarded", "cards_milled",
                "library_events"} <= player
        assert {f"mana_delta_{c.lower()}" for c in COLORS} <= player

    def test_every_counter_type_has_a_delta_plus_a_catch_all(self):
        counters = [s for s in PER_ENTITY_FIELDS if s.name.startswith("counters_delta_")]
        assert len(counters) == len(COUNTER_TYPES) + 1


class TestFieldActivation:
    def test_a_stage_one_corpus_trains_the_fields_its_kinds_supervise(self):
        """FR-085: three of eight classes, and the same heads still train."""
        stage_one = frozenset({
            SamplingClass.RESOLUTION_EFFECT, SamplingClass.COMBAT, "resolution-cost",
        })
        active = {s.name for s in active_fields(stage_one, sparse=True)}
        assert "zone_outcome" in active
        assert "life_delta" in active

    def test_a_field_no_present_class_supervises_contributes_nothing(self):
        stage_one = frozenset({SamplingClass.RESOLUTION_EFFECT, SamplingClass.COMBAT})
        active = {s.name for s in active_fields(stage_one, sparse=True)}
        assert "attacker_legal" not in active
        assert "target_legal" not in active

    def test_the_legality_bits_arrive_with_their_own_records(self):
        active = {s.name for s in active_fields(
            frozenset({
                SamplingClass.PLAYABILITY_LEGALITY, SamplingClass.PLAYABILITY_DECISION,
            }),
            sparse=True,
        )}
        assert {"attacker_legal", "blocker_legal", "min_blockers",
                "target_legal"} <= active

    def test_sparse_fields_wait_for_the_curriculum(self):
        classes = frozenset(SAMPLING_CLASSES)
        before = {s.name for s in active_fields(classes, sparse=False)}
        after = {s.name for s in active_fields(classes, sparse=True)}
        assert "keywords_gained" not in before
        assert "keywords_gained" in after
        assert before < after

    def test_dense_fields_train_from_step_zero(self):
        active = {s.name for s in active_fields(frozenset(SAMPLING_CLASSES), sparse=False)}
        assert "damage_taken" in active
        assert "zone_outcome" in active


class TestFieldSpec:
    def test_an_unknown_field_type_is_rejected_by_width(self):
        spec = FieldSpec("bogus", "not-a-type", FieldScope.PLAYER)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unhandled field type"):
            _ = spec.width


class TestValueHeadLayout:
    def test_every_amount_has_an_at_least_block_and_no_cost_flag_does(self):
        from effects.domain.effect_head_layout import (
            AT_LEAST_SLICES,
            AT_LEAST_STEPS,
            VALUE_SLICES,
            VALUE_WIDTH,
        )
        from effects.domain.value_targets import VALUE_TARGETS, ValueKind

        amounts = {n for n, kind in VALUE_TARGETS if kind is not ValueKind.BINARY}
        assert set(AT_LEAST_SLICES) == amounts
        # The earlier blocks are a prefix: the at-least blocks follow them all.
        first = min(start for start, _ in AT_LEAST_SLICES.values())
        assert first == max(end for _, end in VALUE_SLICES.values())
        assert VALUE_WIDTH == first + len(amounts) * AT_LEAST_STEPS
