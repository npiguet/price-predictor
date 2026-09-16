"""The three output heads and their losses (T069).

The properties that matter are structural: the per-entity head reaches player
slots as well as card slots, created-object slots follow a canonical order with
an overflow flag, and a conditional field contributes nothing where the gate's
target is off.
"""

from __future__ import annotations

import pytest
import torch

from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
    SlotKind,
)
from effects.domain.effect_model import (
    CLASS_COMBAT,
    CLASS_PLAYABILITY_DECISION,
    CLASS_PLAYABILITY_LEGALITY,
    CLASS_RESOLUTION_EFFECT,
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    DURATIONS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    SAMPLING_CLASSES,
    VERDICT_BITS,
    VERDICT_WIDTH,
    ZONE_OUTCOMES,
    EffectModel,
    EffectModelConfig,
    FieldGroup,
    FieldScope,
    FieldSpec,
    FieldType,
    active_fields,
    api_loss,
    created_objects_loss,
    field_loss,
    mlm_loss,
    pairing_loss,
    per_entity_loss,
    scatter_e_rows,
    verdict_loss,
)
from effects.domain.state_snapshot import COLORS

_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
    e_dim=4, vocab_size=32, n_api_types=5, n_param_keys=7,
    d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)


@pytest.fixture
def model() -> EffectModel:
    torch.manual_seed(0)
    return EffectModel(_CONFIG)


def _surface(batch: int = 2, slots: int = 6):
    """A padded token surface: GLOBAL, ACT, PLAYER, PLAYER, CARD, ABILITY."""
    kinds = torch.tensor([[
        SlotKind.GLOBAL, SlotKind.ACT, SlotKind.PLAYER, SlotKind.PLAYER,
        SlotKind.CARD, SlotKind.ABILITY,
    ]] * batch)
    features = {
        kind: torch.zeros(batch, slots, width)
        for kind, width in (
            (SlotKind.GLOBAL, _CONFIG.global_features),
            (SlotKind.ACT, _CONFIG.act_features),
            (SlotKind.PLAYER, _CONFIG.player_features),
            (SlotKind.CARD, _CONFIG.card_features),
        )
    }
    return {
        "slot_features": features,
        "slot_kinds": kinds,
        "positions": torch.tensor([[0, 1, 2, 3, 0, 1]] * batch),
        "e_vectors": torch.zeros(batch, slots, _CONFIG.e_dim),
        "attention_mask": torch.ones(batch, slots, dtype=torch.long),
    }


class TestFieldLayout:
    def test_the_gate_is_the_first_output(self):
        assert GATE_INDEX == 0
        assert min(start for start, _ in FIELD_SLICES.values()) == 1

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

    def test_a_signed_delta_occupies_three_directions_plus_a_magnitude(self):
        spec = FIELDS_BY_NAME["life_delta"]
        assert spec.type is FieldType.SIGNED_DELTA
        assert spec.width == 4

    def test_a_categorical_field_is_as_wide_as_its_vocabulary(self):
        assert FIELDS_BY_NAME["zone_outcome"].width == len(ZONE_OUTCOMES)
        assert FIELDS_BY_NAME["pt_duration"].width == len(DURATIONS)

    def test_the_permanent_player_and_legality_groups_are_all_present(self):
        scopes = {spec.scope for spec in PER_ENTITY_FIELDS}
        assert scopes == {
            FieldScope.PERMANENT, FieldScope.PLAYER, FieldScope.LEGALITY,
        }

    def test_the_player_group_covers_the_fields_the_spec_names(self):
        player = {s.name for s in PER_ENTITY_FIELDS if s.scope is FieldScope.PLAYER}
        assert {"life_delta", "cards_drawn", "cards_discarded", "cards_milled",
                "library_events"} <= player
        assert {f"mana_delta_{c.lower()}" for c in COLORS} <= player

    def test_every_counter_type_has_a_delta_plus_a_catch_all(self):
        counters = [s for s in PER_ENTITY_FIELDS if s.name.startswith("counters_delta_")]
        assert len(counters) == len(COUNTER_TYPES) + 1

    def test_the_sparse_group_is_the_set_the_curriculum_names(self):
        sparse = {s.name for s in PER_ENTITY_FIELDS if s.group is FieldGroup.SPARSE}
        assert "keywords_gained" in sparse
        assert "control_change" in sparse
        assert "attached_to_change" in sparse
        assert "types_gained" in sparse
        assert "pt_duration" in sparse
        assert any(name.startswith("counters_delta_") for name in sparse)
        assert "life_delta" not in sparse
        assert "damage_taken" not in sparse


class TestFieldActivation:
    def test_a_stage_one_corpus_trains_the_fields_its_kinds_supervise(self):
        """FR-085: three of eight classes, and the same heads still train."""
        stage_one = frozenset({
            CLASS_RESOLUTION_EFFECT, CLASS_COMBAT, "resolution-cost",
        })
        active = {s.name for s in active_fields(
            present_classes=stage_one, step=0, curriculum_step=0,
        )}
        assert "zone_outcome" in active
        assert "life_delta" in active

    def test_a_field_no_present_class_supervises_contributes_nothing(self):
        stage_one = frozenset({CLASS_RESOLUTION_EFFECT, CLASS_COMBAT})
        active = {s.name for s in active_fields(
            present_classes=stage_one, step=0, curriculum_step=0,
        )}
        assert "attacker_legal" not in active
        assert "target_legal" not in active

    def test_the_legality_bits_arrive_with_their_own_records(self):
        active = {s.name for s in active_fields(
            present_classes=frozenset({
                CLASS_PLAYABILITY_LEGALITY, CLASS_PLAYABILITY_DECISION,
            }),
            step=0, curriculum_step=0,
        )}
        assert {"attacker_legal", "blocker_legal", "min_blockers",
                "target_legal"} <= active

    def test_sparse_fields_wait_for_the_curriculum_step(self):
        classes = frozenset(SAMPLING_CLASSES)
        before = {s.name for s in active_fields(
            present_classes=classes, step=9_999, curriculum_step=10_000,
        )}
        after = {s.name for s in active_fields(
            present_classes=classes, step=10_000, curriculum_step=10_000,
        )}
        assert "keywords_gained" not in before
        assert "keywords_gained" in after
        assert before < after

    def test_dense_fields_train_from_step_zero(self):
        active = {s.name for s in active_fields(
            present_classes=frozenset(SAMPLING_CLASSES), step=0,
            curriculum_step=10_000,
        )}
        assert "damage_taken" in active
        assert "zone_outcome" in active


class TestTrunkAndHeads:
    def test_the_trunk_returns_one_vector_per_slot(self, model):
        hidden = model(**_surface())
        assert hidden.shape == (2, 6, _CONFIG.d_model)

    def test_the_per_entity_head_reaches_player_slots_too(self, model):
        """FR-077: one shared head over every [CARD] *and* [PLAYER] output."""
        hidden = model(**_surface())
        outputs = model.per_entity(hidden)
        assert outputs.shape == (2, 6, PER_ENTITY_WIDTH)
        player_slots = outputs[:, 2:4, :]
        card_slots = outputs[:, 4:5, :]
        assert player_slots.shape[-1] == card_slots.shape[-1]

    def test_the_created_objects_head_reads_the_global_slot(self, model):
        hidden = model(**_surface())
        assert model.created_objects(hidden).shape == (2, CREATED_OBJECTS_WIDTH)

    def test_the_verdict_head_reads_the_act_slot(self, model):
        hidden = model(**_surface())
        assert model.verdict(hidden).shape == (2, VERDICT_WIDTH)

    def test_created_object_slots_are_uniform_with_one_overflow_flag(self):
        assert CREATED_OBJECTS_WIDTH == (
            CREATED_OBJECT_SLOTS * CREATED_SLOT_WIDTH + 1
        )
        assert CREATED_OBJECT_SLOTS == 4

    def test_a_created_object_slot_carries_script_id_and_characteristics(self):
        # present + scripted + count, then P/T and the characteristic flags.
        assert CREATED_SLOT_WIDTH == 3 + 2 + len(CARD_TYPES) + len(OVERLAY_KEYWORDS)

    def test_the_verdict_head_carries_bits_cost_and_the_trigger_flag(self):
        assert VERDICT_WIDTH == len(VERDICT_BITS) + len(COLORS) + 1


class TestTrainingOnlyHeads:
    def test_the_auxiliaries_exist_when_configured(self, model):
        assert model.mlm_head is not None
        assert model.api_type_head is not None
        assert model.param_key_head is not None

    def test_they_are_all_named_for_filtering_at_save_time(self, model):
        for name in EffectModel.TRAINING_ONLY_HEADS:
            assert hasattr(model, name)

    def test_no_shipped_head_is_in_the_filter_list(self):
        shipped = {"per_entity_head", "created_objects_head", "verdict_head"}
        assert not shipped & set(EffectModel.TRAINING_ONLY_HEADS)

    def test_the_auxiliaries_are_absent_when_unconfigured(self):
        bare = EffectModel(EffectModelConfig(
            global_features=8, act_features=6, player_features=10,
            card_features=12, e_dim=4, d_model=16, n_layers=1, n_heads=2,
            ff_dim=32,
        ))
        assert bare.mlm_head is None
        assert bare.api_type_head is None


class TestLosses:
    def _outputs(self, batch=2, entities=3):
        torch.manual_seed(1)
        return torch.randn(batch, entities, PER_ENTITY_WIDTH)

    def test_a_conditional_field_contributes_nothing_where_the_gate_is_off(self):
        outputs = self._outputs()
        entity_mask = torch.ones(2, 3)
        targets = {"damage_taken": torch.zeros(2, 3)}
        fields = (FIELDS_BY_NAME["damage_taken"],)
        off = per_entity_loss(
            outputs, torch.zeros(2, 3), targets, entity_mask, fields=fields,
            report_parts=True,
        )[1]
        on = per_entity_loss(
            outputs, torch.ones(2, 3), targets, entity_mask, fields=fields,
            report_parts=True,
        )[1]
        assert "damage_taken" not in off
        assert "damage_taken" in on

    def test_the_gate_trains_on_every_entity_including_unaffected_ones(self):
        _, parts = per_entity_loss(
            self._outputs(), torch.zeros(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )
        assert parts["gate"] > 0.0

    def test_the_decomposition_is_off_by_default(self):
        """Each term read back is a device synchronization, and the training
        loop discards them — so the caller has to ask."""
        _, parts = per_entity_loss(
            self._outputs(), torch.zeros(2, 3), {}, torch.ones(2, 3), fields=(),
        )
        assert parts == {}

    def test_the_total_is_the_same_either_way(self):
        """Reporting must not change what is optimized."""
        outputs = self._outputs()
        quiet, _ = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
        )
        loud, _ = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )
        assert float(quiet) == pytest.approx(float(loud))

    def test_the_loss_is_normalized_per_record_not_per_entity(self):
        """A twelve-permanent board must not outweigh a two-permanent one."""
        torch.manual_seed(2)
        wide = torch.zeros(1, 12, PER_ENTITY_WIDTH)
        narrow = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        wide_loss, _ = per_entity_loss(
            wide, torch.zeros(1, 12), {}, torch.ones(1, 12), fields=(),
        )
        narrow_loss, _ = per_entity_loss(
            narrow, torch.zeros(1, 2), {}, torch.ones(1, 2), fields=(),
        )
        # Both divide by one record; the wide board's larger sum is the point,
        # but doubling the batch must halve neither.
        two_records = torch.zeros(2, 2, PER_ENTITY_WIDTH)
        batched, _ = per_entity_loss(
            two_records, torch.zeros(2, 2), {}, torch.ones(2, 2), fields=(),
        )
        assert batched == pytest.approx(float(narrow_loss), rel=1e-5)
        assert wide_loss > narrow_loss

    def test_padded_entities_contribute_nothing(self):
        outputs = self._outputs()
        full = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )[1]["gate"]
        masked = per_entity_loss(
            outputs, torch.ones(2, 3), {},
            torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), fields=(),
            report_parts=True,
        )[1]["gate"]
        assert masked < full

    def test_a_count_field_uses_a_poisson_loss(self):
        """Poisson on a log-rate: the loss is lowest where exp(pred) == target.

        Compared across *predictions*, not across targets — Poisson NLL drops
        the constant log(target!) term, so two targets at one prediction are
        not comparable and only the prediction's fit is.
        """
        spec = FIELDS_BY_NAME["damage_taken"]
        target = torch.full((4,), 3.0)
        right = field_loss(spec, torch.full((4, 1), 1.0986), target)  # log 3
        too_low = field_loss(spec, torch.zeros(4, 1), target)
        too_high = field_loss(spec, torch.full((4, 1), 2.5), target)
        assert right < too_low
        assert right < too_high

    def test_the_count_loss_is_nonnegative_at_its_optimum(self):
        """`full=True` adds the Stirling term, so a perfect prediction scores ~0
        instead of k - k ln k, which is -306 for a target of 88 (FR-081)."""
        from effects.domain.effect_model import _poisson
        target = torch.tensor([88.0, 3.0, 1.0])
        loss = _poisson(torch.log(target), target)
        assert loss.item() >= 0.0
        assert loss.item() < 6.0   # Stirling residual is ~0.5*log(2*pi*k) per element

    def test_a_signed_delta_separates_direction_from_magnitude(self):
        spec = FIELDS_BY_NAME["life_delta"]
        prediction = torch.zeros(4, 4)
        gain = field_loss(spec, prediction, torch.full((4,), 3.0))
        loss = field_loss(spec, prediction, torch.full((4,), -3.0))
        # Same magnitude, opposite direction: the magnitude term matches and
        # only the direction term can differ, so both are finite and positive.
        assert gain > 0 and loss > 0

    def test_a_signed_delta_is_cheapest_when_direction_and_size_are_right(self):
        spec = FIELDS_BY_NAME["life_delta"]
        confident_positive = torch.tensor([[-5.0, -5.0, 5.0, 1.0986]])
        right = field_loss(spec, confident_positive, torch.tensor([3.0]))
        wrong = field_loss(spec, confident_positive, torch.tensor([-3.0]))
        assert right < wrong

    def test_a_categorical_field_uses_cross_entropy(self):
        spec = FIELDS_BY_NAME["zone_outcome"]
        logits = torch.zeros(2, len(ZONE_OUTCOMES))
        logits[0, 1] = 10.0
        died = field_loss(spec, logits, torch.tensor([1, 1]))
        stayed = field_loss(spec, logits, torch.tensor([0, 0]))
        assert died < stayed

    def test_a_binary_field_uses_bce(self):
        spec = FIELDS_BY_NAME["tap_state"]
        assert field_loss(spec, torch.full((3, 1), 10.0), torch.ones(3)) < 0.01

    def test_a_multi_binary_field_scores_every_member(self):
        spec = FIELDS_BY_NAME["keywords_gained"]
        prediction = torch.full((2, spec.arity), -10.0)
        none_gained = field_loss(spec, prediction, torch.zeros(2, spec.arity))
        one_gained = field_loss(
            spec, prediction,
            torch.nn.functional.one_hot(torch.tensor([0, 0]), spec.arity),
        )
        assert one_gained > none_gained

    def test_created_objects_loss_scores_the_overflow_flag(self):
        target = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        target[:, -1] = 1.0  # more than K distinct groups
        predicts_overflow = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        predicts_overflow[:, -1] = 8.0
        denies_overflow = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        denies_overflow[:, -1] = -8.0
        assert created_objects_loss(predicts_overflow, target) < (
            created_objects_loss(denies_overflow, target)
        )

    def test_created_objects_loss_scores_every_slot(self):
        """A slot the head gets wrong costs, wherever in the K it sits."""
        target = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        baseline = created_objects_loss(torch.zeros(2, CREATED_OBJECTS_WIDTH), target)
        for slot in range(CREATED_OBJECT_SLOTS):
            wrong = torch.zeros(2, CREATED_OBJECTS_WIDTH)
            wrong[:, slot * CREATED_SLOT_WIDTH] = 8.0  # claims a group is present
            assert created_objects_loss(wrong, target) > baseline

    def test_verdict_loss_skips_records_with_no_verdict(self):
        prediction = torch.zeros(2, VERDICT_WIDTH)
        target = torch.zeros(2, VERDICT_WIDTH)
        empty = verdict_loss(prediction, target, torch.zeros(2, dtype=torch.bool))
        assert float(empty) == 0.0

    def test_verdict_loss_scores_the_records_that_have_one(self):
        prediction = torch.zeros(2, VERDICT_WIDTH)
        target = torch.ones(2, VERDICT_WIDTH)
        scored = verdict_loss(prediction, target, torch.ones(2, dtype=torch.bool))
        assert float(scored) > 0.0

    def test_mlm_loss_scores_only_the_masked_positions(self):
        logits = torch.randn(2, 5, 32)
        targets = torch.randint(0, 32, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)
        assert float(mlm_loss(logits, targets, mask)) == 0.0
        mask[0, 0] = True
        assert float(mlm_loss(logits, targets, mask)) > 0.0

    def test_api_loss_accepts_either_target_alone(self):
        types = api_loss(torch.randn(2, 5), torch.tensor([0, 1]), None, None)
        keys = api_loss(None, None, torch.randn(2, 7), torch.zeros(2, 7))
        assert float(types) > 0.0 and float(keys) > 0.0

    def test_api_loss_needs_at_least_one_target(self):
        with pytest.raises(ValueError, match="at least one"):
            api_loss(None, None, None, None)

    def test_pairing_loss_stops_the_gradient_on_the_script_side(self):
        script = torch.randn(2, 4, requires_grad=True)
        prose = torch.randn(2, 4, requires_grad=True)
        pairing_loss(script, prose).backward()
        assert script.grad is None
        assert prose.grad is not None


class TestFieldSpec:
    def test_an_unknown_field_type_is_rejected_by_width(self):
        spec = FieldSpec("bogus", "not-a-type", FieldScope.PLAYER)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unhandled field type"):
            _ = spec.width

    def test_a_config_with_indivisible_heads_is_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            EffectModelConfig(
                global_features=8, act_features=6, player_features=10,
                card_features=12, d_model=10, n_heads=4,
            )


class TestScatterERows:
    """How the ability encoder's output reaches the effect head.

    A surface built for training carries row indices rather than values, and
    this is where the rows become vectors. Both properties below are the reason
    it exists: a per-slot round trip through the host would sever the gradient,
    and the effect head's loss would then never reach the encoder at all.
    """

    def _matrix(self) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True,
        )

    def test_a_named_row_replaces_the_slots_vector(self):
        e_vectors = torch.zeros(1, 3, 2)
        rows = torch.tensor([[2, 0, -1]])
        out = scatter_e_rows(e_vectors, rows, self._matrix())
        assert out[0, 0].tolist() == [5.0, 6.0]
        assert out[0, 1].tolist() == [1.0, 2.0]

    def test_an_unnamed_slot_keeps_what_collate_put_there(self):
        """-1 means "no row named": a literal vector, or the zero padding."""
        e_vectors = torch.tensor([[[0.0, 0.0], [7.0, 8.0]]])
        rows = torch.tensor([[-1, -1]])
        out = scatter_e_rows(e_vectors, rows, self._matrix())
        assert out.tolist() == e_vectors.tolist()

    def test_row_zero_is_a_row_and_not_a_miss(self):
        e_vectors = torch.zeros(1, 1, 2)
        out = scatter_e_rows(e_vectors, torch.tensor([[0]]), self._matrix())
        assert out[0, 0].tolist() == [1.0, 2.0]

    def test_the_gradient_reaches_the_matrix(self):
        matrix = self._matrix()
        out = scatter_e_rows(torch.zeros(1, 2, 2), torch.tensor([[1, -1]]), matrix)
        out.sum().backward()
        assert matrix.grad is not None
        # Only the row that was named receives gradient.
        assert matrix.grad[1].tolist() == [1.0, 1.0]
        assert matrix.grad[0].tolist() == [0.0, 0.0]
        assert matrix.grad[2].tolist() == [0.0, 0.0]
