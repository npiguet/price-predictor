"""Where the per-entity head's gate and fields train (T122, T123).

Root spec section 11.4: the legality outputs train on every entity they name,
whatever the affected gate's target, and a playability record trains no gate.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch.nn import functional

from effects.domain.effect_head_input import Slot, SlotKind
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
)
from effects.domain.effect_losses import per_entity_loss
from effects.domain.effect_targets import derive_targets
from effects.domain.surface_collation import entity_target_tensors
from tests.unit.effects.builders import fields_everywhere, make_entity_targets

_RUN = "e8aa2e54-4112-425c-a7fa-96c7bb7340db"
_BERSERK = f"{_RUN}.5-mvyv9f2emq04.33592"


@functools.cache
def _fixture() -> dict:
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return {record.record_id: record for record in read_shard(GEN3_RECORDS)}


@dataclass
class _Surface:
    slots: tuple
    entity_slots: tuple


def _surface_of(record) -> _Surface:
    """One slot per player and per card, in snapshot order."""
    slots = [Slot(SlotKind.GLOBAL, 0), Slot(SlotKind.ACT, 1)]
    for player in record.state.players:
        slots.append(Slot(SlotKind.PLAYER, len(slots), player_id=player.id))
    for entity in record.state.entities:
        slots.append(Slot(SlotKind.CARD, len(slots), entity_id=entity.id))
    return _Surface(tuple(slots), tuple(range(2, len(slots))))


class TestTheGateMask:
    def test_a_playability_record_trains_no_gate_and_an_effect_record_does(self):
        decision = _fixture()[_BERSERK]
        resolution = _fixture()[f"{_RUN}.1-mvyw1pozylvp.3920"]
        records = [decision, resolution]
        batch = entity_target_tensors(
            [_surface_of(r) for r in records], [derive_targets(r) for r in records],
            PER_ENTITY_FIELDS,
        )
        assert batch.gated.tolist() == [False, True]
        assert not batch.gate_scored[0].any()
        assert batch.gate_scored[1].equal(batch.real[1])

    def test_target_legal_trains_on_every_entity_slot_of_the_decision(self):
        """Story 5 acceptance 3, as the loss reads it: scored on every card
        and player slot, none of them affected."""
        record = _fixture()[_BERSERK]
        surface = _surface_of(record)
        batch = entity_target_tensors(
            [surface], [derive_targets(record)], (FIELDS_BY_NAME["target_legal"],),
        )
        assert not batch.affected.any()
        assert batch.scored("target_legal")[0].sum() == len(surface.entity_slots)
        legal = set(record.payload.candidates[0].legal_targets)
        values = batch.fields.values["target_legal"][0]
        for column, slot in enumerate(surface.entity_slots):
            subject = surface.slots[slot].subject_id
            assert values[column] == (1.0 if subject in legal else 0.0), subject

    def test_the_gate_term_skips_an_ungated_record(self):
        torch.manual_seed(0)
        outputs = torch.randn(2, 3, PER_ENTITY_WIDTH)
        gate = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targets = make_entity_targets(gate)
        targets = type(targets)(
            targets.gate, targets.fields, targets.mask, targets.index,
            torch.tensor([True, False]),
        )
        loss, parts = per_entity_loss(outputs, targets, fields=(), report_parts=True)
        expected = functional.binary_cross_entropy_with_logits(
            outputs[0, :, GATE_INDEX], gate[0], reduction="sum",
        ) / 2
        assert torch.allclose(loss, expected)
        assert parts.as_dict()["gate"] == float(expected)


class TestGateFreeFields:
    def test_a_legality_field_trains_where_the_gate_target_is_off(self):
        gate = torch.zeros(1, 3)
        values = {"target_legal": torch.tensor([[1.0, 0.0, 1.0]]),
                  "damage_taken": torch.zeros(1, 3)}
        targets = make_entity_targets(gate, fields_everywhere(values))
        assert targets.scored("target_legal").all()
        assert not targets.scored("damage_taken").any()

    def test_every_legality_field_is_gate_free_and_no_other_is(self):
        free = {spec.name for spec in PER_ENTITY_FIELDS if not spec.gated}
        assert free == {"target_legal", "attacker_legal", "blocker_legal", "min_blockers",
                        "attack_target_legal", "combat_requirement"}

    def test_the_legality_terms_score_the_unaffected_entities(self):
        torch.manual_seed(1)
        outputs = torch.randn(1, 3, PER_ENTITY_WIDTH)
        values = {"attack_target_legal": torch.tensor([[1.0, 0.0, 1.0]])}
        targets = make_entity_targets(torch.zeros(1, 3), fields_everywhere(values))
        spec = FIELDS_BY_NAME["attack_target_legal"]
        _, parts = per_entity_loss(outputs, targets, fields=(spec,), report_parts=True)
        start, _ = FIELD_SLICES[spec.name]
        expected = functional.binary_cross_entropy_with_logits(
            outputs[0, :, start], values["attack_target_legal"][0], reduction="sum",
        )
        assert abs(parts.as_dict()["attack_target_legal"] - float(expected)) < 1e-5
