"""A face-down card's real subtypes ride the subtype table, marked apart (finding 4.8, G05d).

The card slot lists its own subtypes, its chosen types and, on a face-down
card, its real face's subtypes: three id lists over one table. The real-face
rows each take one learned vector, so a real face's ``human`` and a card's own
``human`` are two inputs.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.input_layout import SlotKind
from effects.domain.input_vocabularies import ID_PADDING, subtype_id
from effects.domain.slot_widths import layout_widths
from effects.domain.surface_builder import SurfaceBuilder
from effects.domain.surface_collation import collate_surfaces
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: E35 is face down with a bird soldier warrior real face (a record of the fixture).
_RECORD = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv.2133"
_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
    ability_features=7, e_dim=4, vocab_size=32, n_api_types=5, n_param_keys=7,
    d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)


@pytest.fixture(scope="module")
def record():
    found = next(found for found in map(record_from_dict, gen3_fixture_records())
                 if found.record_id == _RECORD)
    card = found.state.entity("E35")
    toy = replace(card, subtypes=("toy",))
    return replace(found, state=replace(found.state, entities=tuple(
        toy if entity.id == "E35" else entity for entity in found.state.entities)))


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def _card_slot(record, entity_id):
    return next(slot for slot in _surface(record).slots
                if slot.kind is SlotKind.CARD and slot.entity_id == entity_id)


def _ids(row):
    return tuple(int(i) for i in row if int(i) != ID_PADDING)


def test_the_slot_lists_the_own_and_the_real_face_rows_apart(record):
    slot = _card_slot(record, "E35")
    assert slot.subtypes == (subtype_id("toy"),)
    assert slot.real_face_subtypes == tuple(
        subtype_id(name) for name in ("bird", "soldier", "warrior"))


def test_a_card_with_no_real_face_lists_none(record):
    other = next(entity.id for entity in record.state.entities
                 if entity.real_face is None and entity.subtypes)
    assert _card_slot(record, other).real_face_subtypes == ()


def test_the_collation_pads_the_real_face_list_like_the_others(record):
    surface = _surface(record)
    inputs = collate_surfaces([surface], e_dim=E_DIM, widths=layout_widths())
    column = next(i for i, slot in enumerate(surface.slots) if slot.entity_id == "E35"
                  and slot.kind is SlotKind.CARD)
    assert _ids(inputs.real_face_subtype_ids[0, column]) == _card_slot(
        record, "E35").real_face_subtypes
    assert _ids(inputs.subtype_ids[0, column]) == (subtype_id("toy"),)
    assert inputs.real_face_subtype_ids.shape == inputs.subtype_ids.shape


def test_the_card_sum_adds_the_offset_once_per_real_face_row(record):
    surface = _surface(record)
    inputs = collate_surfaces([surface], e_dim=E_DIM, widths=layout_widths())
    column = next(i for i, slot in enumerate(surface.slots) if slot.entity_id == "E35"
                  and slot.kind is SlotKind.CARD)
    torch.manual_seed(0)
    model = EffectModel(_CONFIG)
    with torch.no_grad():
        marked = model._card_ids(inputs)
        offset = model.real_face_subtype_offset.clone()
        model.real_face_subtype_offset.zero_()
        unmarked = model._card_ids(inputs)
    assert torch.allclose(marked[0, column] - unmarked[0, column], 3 * offset, atol=1e-5)
    # Every other column, a card with no real face included, reads as before.
    other = torch.tensor([not slot.real_face_subtypes for slot in surface.slots])
    assert torch.allclose(marked[0, other], unmarked[0, other])


def test_a_checkpoint_saved_before_the_offset_existed_is_refused():
    from types import SimpleNamespace

    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.infrastructure.effect_model_store import load_weights

    state = EffectModel(_CONFIG).state_dict()
    del state["real_face_subtype_offset"]
    encoder = AbilityEncoderConfig(vocab_size=32, e_dim=4, d_model=16, n_layers=1, n_heads=2)
    old = SimpleNamespace(
        encoder_config=encoder, encoder_state=AbilityEncoder(encoder).state_dict(),
        model_config=_CONFIG, model_state=state)
    with pytest.raises(RuntimeError, match="real_face_subtype_offset"):
        load_weights(old, "cpu")
