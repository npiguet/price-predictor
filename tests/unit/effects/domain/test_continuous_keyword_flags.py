"""A continuous record's keyword flags read the board as written.

The collector writes a continuous record's board without the static the record
is about, so a keyword the creature holds on that board comes from another
source, and its flag stays set even where the static grants that keyword too.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.effect_head_input import SlotKind, SurfaceBuilder, card_segments
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.records import RecordKind
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: Kitesail equipped to Summit Apes (E22): the static grants flying.
_KITESAIL = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.118"


@pytest.fixture(scope="module")
def kitesail():
    return next(record for record in map(record_from_dict, gen3_fixture_records())
                if record.record_id == _KITESAIL)


def _flying(record, entity_id: str) -> float:
    """The flying flag of ``entity_id``'s card slot on the record's surface."""
    surface = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)
    slot = next(slot for slot in surface.slots if slot.entity_id == entity_id
                and slot.kind is SlotKind.CARD)
    entity = next(entity for entity in record.state.entities if entity.id == entity_id)
    flags = slot.features[card_segments(entity, record)["keyword_flags"]]
    return flags[KEPT_KEYWORD_FLAGS.index("flying")]


def test_the_fixture_record_is_kitesails_flying_grant(kitesail):
    assert kitesail.kind is RecordKind.CONTINUOUS
    assert [(c.entity, c.keywords) for c in kitesail.payload.contributions
            if c.keywords] == [("E22", ("flying",))]


def test_a_keyword_held_from_another_source_keeps_its_flag(kitesail):
    """Summit Apes flies from another source too (a Jump, say): the board
    holds flying, and the flag says so."""
    apes = next(entity for entity in kitesail.state.entities if entity.id == "E22")
    flying_apes = replace(apes, granted_temporary=replace(
        apes.granted_temporary, keywords=("Flying",)))
    record = replace(kitesail, state=replace(kitesail.state, entities=tuple(
        flying_apes if entity.id == "E22" else entity for entity in kitesail.state.entities)))
    assert _flying(kitesail, "E22") == 0.0
    assert _flying(record, "E22") == 1.0
