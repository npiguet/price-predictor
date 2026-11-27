"""A combat record's damage assignment is no input.

The assignment states who deals damage in the step and how much, so it reads
the first-strike rule without the keyword. The surface takes nothing from it:
two combat records that differ only in their ``assignment_choices`` build the
same surface. Who attacks and who blocks stays readable from the board.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.card_features import card_segments
from effects.domain.input_layout import SlotKind
from effects.domain.records import RecordKind
from effects.domain.surface_builder import SurfaceBuilder
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: E43 attacks and is blocked by E27 and E22, splitting its damage 2 and 2.
_DOUBLE_BLOCK = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.240"


@pytest.fixture(scope="module")
def double_block():
    return next(record for record in map(record_from_dict, gen3_fixture_records())
                if record.record_id == _DOUBLE_BLOCK)


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def _assigned(record, choices):
    return replace(record, payload=replace(record.payload, assignment_choices=choices))


def _combat_status(record, entity_id: str):
    """The attacking, blocking, blocked and blocked-by inputs of a card slot."""
    slot = next(slot for slot in _surface(record).slots
                if slot.entity_id == entity_id and slot.kind is SlotKind.CARD)
    status = slot.features[card_segments(record.state.entity(entity_id), record)["status"]]
    return status[-5:-2]


def test_the_fixture_record_is_a_double_block_with_a_split(double_block):
    assert double_block.kind is RecordKind.COMBAT
    assert double_block.payload.blocks == {"E43": ("E27", "E22")}
    assert double_block.payload.assignment_choices["E43"] == {"E22": 2, "E27": 2}


@pytest.mark.parametrize("choices", [
    pytest.param({"E43": {"E22": 4}, "E27": {"E43": 2}, "E22": {"E43": 6}}, id="other-split"),
    pytest.param({}, id="no-assignment"),
])
def test_records_differing_only_in_the_assignment_build_equal_surfaces(
        double_block, choices):
    assert _surface(_assigned(double_block, choices)) == _surface(double_block)


def test_the_board_still_says_who_attacks_and_who_blocks(double_block):
    record = _assigned(double_block, {})
    assert _combat_status(record, "E43")[0] == 1.0
    assert _combat_status(record, "E43")[2] == 1.0
    assert _combat_status(record, "E27")[1] == 1.0
