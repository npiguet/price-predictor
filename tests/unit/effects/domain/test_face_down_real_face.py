"""A face-down card carries its real face as inactive ability slots (finding 4.8, S15 answer 7).

What turning a card face up does depends on its real face. The collector
writes a face-down card's real-face lines among its ``lines`` with
``active = False`` (``CardLine``), so they reach the surface as ability
slots flagged inactive, each with its line's ``e``; the card slot reads
``real_face``'s characteristics in a part of its own.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.effect_head_input import (
    SlotKind,
    SurfaceBuilder,
    card_features,
    card_segments,
    face_state_of,
    slot_lines,
)
from effects.domain.state_snapshot import FaceState
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: Antagonize, E22, exiled face down by Rabble Rousing's trigger.
_EXILED_FACE_DOWN = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv.1410"


@pytest.fixture(scope="module")
def record():
    return next(found for found in map(record_from_dict, gen3_fixture_records())
                if found.record_id == _EXILED_FACE_DOWN)


def _with_entity(record, entity):
    return replace(record, state=replace(record.state, entities=tuple(
        entity if found.id == entity.id else found for found in record.state.entities)))


def _ability_slots(surface, entity_id: str):
    return [slot for slot in surface.slots
            if slot.kind is SlotKind.ABILITY and slot.entity_id == entity_id]


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def test_the_real_faces_lines_are_written_inactive(record):
    card = record.state.entity("E22")
    assert face_state_of(card) == FaceState.FACE_DOWN
    assert card.real_face.name == "Antagonize"
    assert card.lines and all(line.active is False for line in card.lines)
    assert [line.active for line in slot_lines(card)] == [False] * len(card.lines)


def test_the_real_faces_lines_are_ability_slots_flagged_inactive(record):
    slots = _ability_slots(_surface(record), "E22")
    assert len(slots) == len(record.state.entity("E22").lines)
    assert all(slot.features[0] == 0.0 for slot in slots)


def test_a_face_down_card_with_another_real_face_reads_apart(record):
    card = record.state.entity("E22")
    blank = _with_entity(record, replace(card, lines=()))
    assert _ability_slots(_surface(blank), "E22") == []
    assert _surface(record).slots != _surface(blank).slots


def _real_face_part(record, entity):
    return card_features(entity, record)[card_segments(entity, record)["real_face"]]


def _changed(record, **changes):
    card = record.state.entity("E22")
    renamed = replace(card, real_face=replace(card.real_face, **changes))
    return _with_entity(record, renamed), renamed


@pytest.mark.parametrize("changes", [
    {"base_pt": (3, 3)},
    {"types": ("creature",)},
    {"colors": ("U",)},
    {"mana_value": 5},
    {"subtypes": ("goblin", "wizard")},
])
def test_each_characteristic_of_the_real_face_reaches_the_card_slot(record, changes):
    other, card = _changed(record, **changes)
    assert _real_face_part(record, record.state.entity("E22")) != _real_face_part(other, card)
    assert _surface(record).slots != _surface(other).slots


def test_two_face_down_records_differing_only_in_the_real_faces_power_differ(record):
    low, _ = _changed(record, base_pt=(1, 2))
    high, _ = _changed(record, base_pt=(4, 2))
    assert _surface(low).slots != _surface(high).slots


def test_the_face_down_characteristics_stay_where_they_are(record):
    other, card = _changed(record, mana_value=5, types=("creature",))
    whole = card_segments(card, other)["characteristics"]
    assert card_features(card, other)[whole] == card_features(
        record.state.entity("E22"), record)[whole]


def test_a_card_that_is_not_face_down_reads_zeros_there(record):
    card = record.state.entity("E1")
    assert card.real_face is None
    assert not any(_real_face_part(record, card))
