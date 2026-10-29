"""Which fields an affected entity is scored on (FR-077, FR-081).

A record that says nothing about a field must not be read as having said zero:
an untapped creature is not one no event tapped, and a creature that was
destroyed carries no tap state at all. These go through the public pipeline —
targets, collation, loss — so they hold whatever the targets look like inside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    PER_ENTITY_WIDTH,
)
from effects.domain.effect_losses import per_entity_loss
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import PlayerState
from effects.domain.surface_collation import entity_target_tensors
from tests.unit.effects.builders import (
    make_entity,
    make_entity_targets,
    make_record,
    make_snapshot,
)


@dataclass
class _Slot:
    kind: SlotKind
    entity_id: str | None = None
    player_id: str | None = None


@dataclass
class _Surface:
    slots: tuple
    entity_slots: tuple


_SURFACE = _Surface(
    slots=(
        _Slot(SlotKind.GLOBAL), _Slot(SlotKind.ACT),
        _Slot(SlotKind.PLAYER, player_id="P0"),
        _Slot(SlotKind.CARD, entity_id="E1"),
    ),
    entity_slots=(2, 3),
)
_SNAPSHOT = make_snapshot(
    players=(PlayerState(id="P0", life=20, hand=7, library=40, graveyard=0),),
    entities=(make_entity("E1", "Bear"),),
)


def _resolution(*events):
    return make_record(state=_SNAPSHOT, payload=ResolutionPayload(events=events))


def _losses(record, names, outputs):
    fields = tuple(FIELDS_BY_NAME[name] for name in names)
    targets = entity_target_tensors([_SURFACE], [derive_targets(record)], fields)
    return per_entity_loss(outputs, targets, fields=fields, report_parts=True)


def _moves(record, scored: str, perturbed: str | None = None) -> bool:
    """Whether pushing ``perturbed``'s outputs moves the loss of ``scored``."""
    perturbed = perturbed or scored
    torch.manual_seed(0)
    outputs = torch.randn(1, 2, PER_ENTITY_WIDTH)
    names = sorted({scored, perturbed, "zone_outcome"})
    before = _losses(record, names, outputs)[0]
    start, end = FIELD_SLICES[perturbed]
    pushed = outputs.clone()
    # A ramp rather than a constant: a constant shift leaves a softmax as it was.
    pushed[..., start:end] += 3.0 * torch.arange(1, end - start + 1)
    after = _losses(record, names, pushed)[0]
    return abs(float(after) - float(before)) > 1e-4


_DESTROYED = Event(type=EventType.DESTROYED, subjects=("E1",))
_LIFE_LOST = Event(type=EventType.LIFE_CHANGE, subjects=("P0",), params={"delta": -3})


class TestAbsentFieldsAreNotScored:
    def test_an_absent_field_on_an_affected_entity_contributes_no_loss(self):
        """A destroyed creature had no tap event: its tap state is unknown."""
        assert not _moves(_resolution(_DESTROYED), "tap_state")

    def test_a_tap_event_is_scored(self):
        tapped = Event(type=EventType.TAPPED, subjects=("E1",))
        assert _moves(_resolution(tapped), "tap_state")

    def test_an_untap_event_is_scored_as_untapped(self):
        untapped = Event(type=EventType.UNTAPPED, subjects=("E1",))
        assert _moves(_resolution(untapped), "tap_state")

    def test_a_duration_no_event_states_is_not_scored(self):
        pumped = Event(
            type=EventType.PT_CHANGE, subjects=("E1",),
            params={"power_delta": 2, "toughness_delta": 2},
        )
        assert not _moves(_resolution(pumped), "pt_duration")

    def test_a_player_is_not_scored_on_a_permanents_fields(self):
        assert not _moves(_resolution(_LIFE_LOST), "zone_outcome")
        assert not _moves(_resolution(_LIFE_LOST), "damage_taken")

    def test_a_permanent_is_not_scored_on_a_players_fields(self):
        assert not _moves(_resolution(_DESTROYED), "life_delta")
        assert not _moves(_resolution(_DESTROYED), "mana_delta_r")


class TestSilentFieldsReadTheirKnownValue:
    def test_an_affected_permanent_that_took_no_damage_is_scored_on_zero_damage(self):
        assert _moves(_resolution(_DESTROYED), "damage_taken")

    def test_a_damaged_permanent_that_did_not_move_is_scored_as_stayed(self):
        hit = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 1})
        assert _moves(_resolution(hit), "zone_outcome")

    def test_an_affected_player_who_drew_nothing_is_scored_on_zero_cards(self):
        assert _moves(_resolution(_LIFE_LOST), "cards_drawn")


class TestRecordKindsSuperviseTheirOwnFields:
    def test_a_legality_record_says_nothing_about_zones(self):
        record = make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
            state=_SNAPSHOT,
            payload=PlayabilityAttackersPayload(
                forbidden=(ForbiddenEntity(entity="E1"),),
            ),
        )
        assert _moves(record, "attacker_legal")
        assert not _moves(record, "damage_taken")
        assert not _moves(record, "zone_outcome")

    def test_a_static_that_sets_a_type_line_says_nothing_about_what_it_removed(self):
        record = make_record(
            RecordKind.CONTINUOUS, state=_SNAPSHOT,
            payload=ContinuousPayload(contributions=(
                Contribution(entity="E1", types=("=", "artifact")),
            )),
        )
        assert _moves(record, "types_gained")
        assert not _moves(record, "types_lost")

    def test_a_static_that_only_grants_a_keyword_is_scored_on_no_pt_change(self):
        record = make_record(
            RecordKind.CONTINUOUS, state=_SNAPSHOT,
            payload=ContinuousPayload(contributions=(
                Contribution(entity="E1", keywords=("flying",)),
            )),
        )
        assert _moves(record, "power_delta")


class TestTheFloorReadsPresenceToo:
    def test_an_absent_target_does_not_move_the_constant_predictor(self):
        """The floor is scored on the entities the loss is, absent ones excluded."""
        from effects.domain.entity_field_targets import EntityFieldTargets
        from effects.domain.loss_floor import constant_predictor_floor

        spec = FIELDS_BY_NAME["tap_state"]
        gate = torch.ones(1, 4)

        def floor(present):
            batch = make_entity_targets(
                gate=gate,
                fields=EntityFieldTargets(
                    {"tap_state": torch.tensor([[1.0, 0.0, 0.0, 0.0]])},
                    {"tap_state": torch.tensor([present])},
                ),
                mask=gate,
            )
            return constant_predictor_floor([batch], fields=(spec,))["tap_state"]

        # Two present entities, one tapped: a coin flip over two entities.
        assert floor([True, True, False, False]) == pytest.approx(2 * math.log(2))
