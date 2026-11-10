"""The verdict, created-objects and global heads' gen-3 outputs (T120, T121, T123).

Root spec section 11.2 puts the outputs on the acting ability on the verdict
head, section 11.5 gives it a ``timing`` column and one per failed check, and
section 11.1 adds a global head beside the created-objects head for what
belongs to no card or player.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import pytest
import torch

from effects.domain.aux_head_targets import (
    cost_adjustment,
    created_objects_targets,
    global_targets,
    mana_symbols,
    verdict_targets,
)
from effects.domain.aux_heads import AUX_HEADS, SHIPPED_HEADS, Head
from effects.domain.effect_head_input import CARD_TYPES, MANA_SYMBOLS
from effects.domain.effect_head_layout import (
    CREATED_KINDS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    DAY_NIGHT_CHANGES,
    FAILED_CHECKS,
    GLOBAL_LAYOUT,
    GLOBAL_WIDTH,
    REPLACEMENT_RESULTS,
    RESOLUTION_OUTCOMES,
    TURN_STRUCTURE_CHANGES,
    VERDICT_BITS,
    VERDICT_LAYOUT,
    VERDICT_WIDTH,
    CreatedSlot,
    RecordField,
)
from effects.domain.effect_losses import global_loss, record_head_loss, verdict_loss
from effects.domain.event_schema import Event, EventType
from effects.domain.field_kinds import FIELD_KINDS
from effects.domain.field_types import FieldType
from effects.domain.output_vocabularies import SUBTYPES_CHANGED
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    PlayabilityCheck,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.state_snapshot import Refs
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

_RUN = "e8aa2e54-4112-425c-a7fa-96c7bb7340db"


@functools.cache
def _fixture() -> dict:
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return {record.record_id: record for record in read_shard(GEN3_RECORDS)}


def _column(row, layout, name):
    """``name``'s target values and whether the row sets it."""
    values, mask = row
    start, end = layout.target_slices[name]
    return values[start:end], mask[layout.index[name]]


class TestTheLayouts:
    def test_every_record_field_owns_a_slice_of_its_width_and_the_slices_tile(self):
        for layout, width in ((VERDICT_LAYOUT, VERDICT_WIDTH), (GLOBAL_LAYOUT, GLOBAL_WIDTH)):
            offset = 0
            for spec in layout.fields:
                assert layout.slices[spec.name] == (offset, offset + spec.width)
                offset += spec.width
            assert offset == width == layout.width

    def test_the_verdict_head_names_every_column_the_spec_lists(self):
        names = [str(spec.name) for spec in VERDICT_LAYOUT.fields]
        assert names[:3] == list(VERDICT_BITS)
        assert {"timing", "failed_checks", "cost_adjustment", "resolution_outcome",
                "effects_created", "replacement_result", "mana_paid",
                "trigger_fired"} <= set(names)

    def test_one_failed_check_column_per_check_the_records_name(self):
        """The columns follow ``PlayabilityCheck``: ``script_restriction`` and
        ``land_play_limit`` need no code of their own (FR-053b)."""
        spec = VERDICT_LAYOUT.by_name["failed_checks"]
        assert spec.members == tuple(check.value for check in PlayabilityCheck)
        assert {"script_restriction", "land_play_limit"} <= set(spec.members)

    def test_the_widths(self):
        assert VERDICT_LAYOUT.by_name["cost_adjustment"].width == 4 * len(MANA_SYMBOLS)
        assert VERDICT_LAYOUT.by_name["resolution_outcome"].width == len(RESOLUTION_OUTCOMES)
        assert VERDICT_LAYOUT.by_name["replacement_result"].width == len(REPLACEMENT_RESULTS)
        assert GLOBAL_LAYOUT.by_name["turn_structure_change"].width == len(
            TURN_STRUCTURE_CHANGES)
        assert GLOBAL_LAYOUT.by_name["day_night_change"].width == len(DAY_NIGHT_CHANGES)

    def test_a_record_field_refuses_a_multi_binary(self):
        with pytest.raises(ValueError):
            RecordField("x", FieldType.MULTI_BINARY, classes=("a", "b"))


class TestTheGlobalHead:
    def test_it_is_a_shipped_row_beside_the_created_objects_head(self):
        heads = [spec.head for spec in AUX_HEADS]
        assert heads.index(Head.GLOBAL) == heads.index(Head.CREATED_OBJECTS) + 1
        assert Head.GLOBAL in {spec.head for spec in SHIPPED_HEADS}

    def test_the_model_reads_it_at_global(self):
        from effects.domain.effect_model import EffectModel, EffectModelConfig

        model = EffectModel(EffectModelConfig(
            global_features=4, act_features=4, player_features=4, card_features=4, e_dim=8,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        ))
        hidden = torch.randn(3, 5, 16)
        assert model.global_outputs(hidden).shape == (3, GLOBAL_WIDTH)
        assert model.verdict(hidden).shape == (3, VERDICT_WIDTH)


class TestVerdictTargets:
    def test_a_candidate_whose_timing_fails_reads_casting_speed(self):
        """Story 5 acceptance 4: ``timing`` 0 and ``casting_speed`` 1."""
        record = _fixture()[f"{_RUN}.0-mvyv5jxepqmy.6"]
        candidate = record.payload.candidates[0]
        assert candidate.timing is False
        row = verdict_targets(record)
        assert _column(row, VERDICT_LAYOUT, "timing") == ([0.0], True)
        failed, supervised = _column(row, VERDICT_LAYOUT, "failed_checks")
        assert supervised
        named = {check.check for check in candidate.failed_checks}
        assert dict(zip(FAILED_CHECKS, failed)) == {
            check: (1.0 if check in named else 0.0) for check in FAILED_CHECKS
        }
        assert failed[FAILED_CHECKS.index("casting_speed")] == 1.0

    def test_a_decision_sets_its_verdict_bits_and_nothing_of_the_other_kinds(self):
        row = verdict_targets(_fixture()[f"{_RUN}.5-mvyv9f2emq04.33592"])
        _, mask = row
        set_fields = {str(spec.name) for spec, on in zip(VERDICT_LAYOUT.fields, mask) if on}
        assert set_fields == {*VERDICT_BITS, "timing", "failed_checks", "cost_adjustment"}

    def test_a_cost_half_sets_the_mana_it_paid_and_no_resolution_outcome(self):
        """The resolution outcome is supervised on effect halves only (FR-052):
        a cost half, even one a pre-moments gen-3 shard stamped, sets none."""
        row = verdict_targets(_fixture()[f"{_RUN}.0-mvyv5jxepqmy.818"])
        assert _column(row, VERDICT_LAYOUT, "resolution_outcome")[1] is False
        paid, supervised = _column(row, VERDICT_LAYOUT, "mana_paid")
        assert supervised and sum(paid) == 5.0

    def test_an_effect_half_sets_its_resolution_outcome(self):
        """FR-052: over resolved, fizzled, partially fizzled and declined."""
        record = _fixture()[f"{_RUN}.0-mvyv84qu3axl.27377"]
        fizzled = replace(record, payload=replace(
            record.payload, events=(), outcome=ResolutionOutcome.FIZZLED))
        assert _column(verdict_targets(fizzled), VERDICT_LAYOUT, "resolution_outcome") == (
            [float(RESOLUTION_OUTCOMES.index("fizzled"))], True)
        assert "countered" not in RESOLUTION_OUTCOMES

    def test_an_effect_half_counts_the_effect_objects_it_created(self):
        row = verdict_targets(_fixture()[f"{_RUN}.0-mvyv84qu3axl.27377"])
        assert _column(row, VERDICT_LAYOUT, "effects_created") == ([1.0], True)

    def test_a_rewrite_supervises_the_replacement_result(self):
        row = verdict_targets(_fixture()[f"{_RUN}.5-mvyvma0up67n.6783"])
        assert _column(row, VERDICT_LAYOUT, "replacement_result") == (
            [float(REPLACEMENT_RESULTS.index("replaced"))], True)

    def test_a_trigger_sets_only_whether_it_fired(self):
        row = verdict_targets(_fixture()[f"{_RUN}.0-mvyv5jxepqmy.127"])
        assert sum(row[1]) == 1
        assert _column(row, VERDICT_LAYOUT, "trigger_fired") == ([1.0], True)


def _decision(charged, printed, *, index=0, zone="hand", land_play=False):
    key = ProvenanceKey("cardsfolder/l/lightning_blast.txt", 0, "spell", index)
    record = make_record(
        RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION,
        payload=PlayabilityDecisionPayload(candidates=(Candidate(
            ability=(key,), can_play=True, affordable=True, has_legal_target=True,
            cost_after_adjustment={"mana": charged}, land_play=land_play,
        ),)),
        state=make_snapshot(
            entities=(make_entity("E1", "Lightning Blast", zone=zone, mana_cost=printed),),
            refs=Refs(source="E1"),
        ),
    )
    return record, record.payload.candidates[0]


class TestCostAdjustment:
    def test_a_reduction_is_a_negative_delta_on_its_symbol(self):
        adjustment = cost_adjustment(*_decision("{1}{R}", {"generic": 3, "R": 1}))
        assert adjustment[MANA_SYMBOLS.index("generic")] == -2.0
        assert adjustment[MANA_SYMBOLS.index("R")] == 0.0
        assert sum(abs(value) for value in adjustment) == 2.0

    def test_the_fixture_spells_cast_at_their_printed_cost(self):
        record = _fixture()[f"{_RUN}.0-mvyv5jxepqmy.2"]
        assert cost_adjustment(record, record.payload.candidates[0]) == [0.0] * len(MANA_SYMBOLS)

    @pytest.mark.parametrize("decision", [
        {"index": 1},
        {"zone": "battlefield"},
        {"land_play": True},
    ], ids=["an-activated-ability", "on-the-battlefield", "a-land-play"])
    def test_only_a_cast_is_measured_against_the_printed_cost(self, decision):
        assert cost_adjustment(*_decision("{R}", {"R": 1}, **decision)) is None

    def test_mana_symbols_reads_forge_costs(self):
        assert mana_symbols("{2}{R}") == {"generic": 2, "R": 1}
        assert mana_symbols("{X}{W/U}{G/P}{0}") == {"X": 1, "W/U": 1, "G/P": 1}
        assert mana_symbols("{H}") is None


def _resolution(*events):
    return make_record(payload=ResolutionPayload(events=tuple(events)))


class TestGlobalTargets:
    def test_turn_structure_and_day_night_changes(self):
        row = global_targets(_resolution(
            Event(type=EventType.TURN_ADDED, subjects=("P0",), params={"count": 2}),
            Event(type=EventType.PHASE_SKIPPED, subjects=("P0",)),
            Event(type=EventType.DAY_NIGHT_CHANGED, params={"to": "night"}),
        ))
        counts, supervised = _column(row, GLOBAL_LAYOUT, "turn_structure_change")
        assert supervised
        assert dict(zip(TURN_STRUCTURE_CHANGES, counts)) == {
            "extra_turns": 2.0, "turns_skipped": 0.0, "extra_phases": 0.0,
            "phases_skipped": 1.0,
        }
        assert _column(row, GLOBAL_LAYOUT, "day_night_change") == (
            [float(DAY_NIGHT_CHANGES.index("became_night"))], True)

    def test_an_ordinary_resolution_changed_neither(self):
        row = global_targets(_fixture()[f"{_RUN}.1-mvyw1pozylvp.3920"])
        assert _column(row, GLOBAL_LAYOUT, "turn_structure_change") == ([0.0] * 4, True)
        assert _column(row, GLOBAL_LAYOUT, "day_night_change") == (
            [float(DAY_NIGHT_CHANGES.index("unchanged"))], True)

    def test_a_decision_supervises_none(self):
        assert global_targets(_fixture()[f"{_RUN}.0-mvyv5jxepqmy.6"]) is None


class TestCreatedObjects:
    def test_a_token_group_carries_its_kind_types_and_subtypes(self):
        target, known = created_objects_targets(_fixture()[f"{_RUN}.1-mvyw1pozylvp.3920"])
        assert len(target) == len(known) == CREATED_OBJECTS_WIDTH
        assert target[CreatedSlot.PRESENT] == 1.0
        assert target[CreatedSlot.KIND + CREATED_KINDS.index("token")] == 1.0
        assert target[CreatedSlot.TYPES + CARD_TYPES.index("artifact")] == 1.0
        assert target[CreatedSlot.SUBTYPES + SUBTYPES_CHANGED.index("treasure")] == 1.0

    def test_what_a_token_event_does_not_say_trains_nothing(self):
        """Neither the controller nor the counters are on ``token_created``."""
        _, known = created_objects_targets(_fixture()[f"{_RUN}.1-mvyw1pozylvp.3920"])
        unknown = {column for column, said in enumerate(known[:CREATED_SLOT_WIDTH]) if not said}
        assert unknown == set(range(CreatedSlot.COUNTERS, CreatedSlot.OTHER_CONTROLLER + 1))
        assert all(known[CREATED_SLOT_WIDTH:])

    def test_a_token_event_naming_its_controller_and_counters_trains_them(self):
        target, known = created_objects_targets(_resolution(Event(
            type=EventType.TOKEN_CREATED,
            params={"count": 1, "token_script_id": "g_x_x_hydra",
                    "characteristics": {"power": 0, "toughness": 0,
                                        "types": "Creature - Hydra",
                                        "controller": "P1", "counters": {"+1/+1": 3}}},
        )))
        assert known[CreatedSlot.OTHER_CONTROLLER] and target[CreatedSlot.OTHER_CONTROLLER] == 1.0
        assert target[CreatedSlot.COUNTERS] == 3.0  # P1P1 leads the counter list

    def test_cards_made_and_effect_objects_are_groups_of_their_own_kinds(self):
        target, _ = created_objects_targets(_resolution(
            Event(type=EventType.CARD_MADE,
                  params={"card_name": "Lightning Bolt", "count": 2, "to_zone": "Hand"}),
            Event(type=EventType.CONTINUOUS_EFFECT_CREATED, params={"layers": "Unblockable"}),
        ))
        kinds = {
            CREATED_KINDS[target[base + CreatedSlot.KIND:base + CreatedSlot.TYPES].index(1.0)]:
                target[base + CreatedSlot.COUNT]
            for base in range(0, CREATED_OBJECTS_WIDTH - 1, CREATED_SLOT_WIDTH)
            if target[base + CreatedSlot.PRESENT]
        }
        assert kinds == {"card": 2.0, "effect": 1.0}


class TestRecordHeadLosses:
    def test_members_score_as_each_member_alone(self):
        generator = torch.Generator().manual_seed(5)
        prediction = torch.randn(3, VERDICT_WIDTH, generator=generator)
        target = torch.zeros(3, VERDICT_LAYOUT.target_width)
        start, end = VERDICT_LAYOUT.target_slices["cost_adjustment"]
        target[:, start:end] = torch.randint(-2, 3, (3, end - start), generator=generator).float()
        mask = torch.zeros(3, len(VERDICT_LAYOUT.fields), dtype=torch.bool)
        mask[:, VERDICT_LAYOUT.index["cost_adjustment"]] = True
        out_start, _ = VERDICT_LAYOUT.slices["cost_adjustment"]
        kind = FIELD_KINDS[FieldType.SIGNED_DELTA]
        dispersion = torch.linspace(-1.0, 2.0, len(VERDICT_LAYOUT.dispersed))
        shared = dispersion[VERDICT_LAYOUT.dispersion_index["cost_adjustment"]]
        expected = sum(
            kind.loss(prediction[:, out_start + 4 * m:out_start + 4 * m + 4],
                      target[:, start + m], arity=1, cap=40.0, log_dispersion=shared)
            for m in range(end - start)
        ) / 3
        assert torch.allclose(
            verdict_loss(prediction, target, mask, log_dispersion=dispersion),
            expected, rtol=1e-5,
        )

    def test_an_unsupervised_batch_scores_nothing(self):
        prediction = torch.randn(2, GLOBAL_WIDTH)
        target = torch.zeros(2, GLOBAL_LAYOUT.target_width)
        mask = torch.zeros(2, len(GLOBAL_LAYOUT.fields), dtype=torch.bool)
        assert float(global_loss(prediction, target, mask)) == 0.0
        assert float(record_head_loss(prediction, target, mask.any(-1), GLOBAL_LAYOUT)) == 0.0

    def test_a_categorical_target_reads_its_class_index(self):
        prediction = torch.zeros(1, GLOBAL_WIDTH)
        start, _ = GLOBAL_LAYOUT.slices["day_night_change"]
        prediction[0, start + 1] = 10.0  # became_night
        target = torch.zeros(1, GLOBAL_LAYOUT.target_width)
        column, _ = GLOBAL_LAYOUT.target_slices["day_night_change"]
        mask = torch.zeros(1, 2, dtype=torch.bool)
        mask[0, GLOBAL_LAYOUT.index["day_night_change"]] = True
        target[0, column] = 1.0
        right = float(global_loss(prediction, target, mask))
        target[0, column] = 0.0
        assert right < float(global_loss(prediction, target, mask))
