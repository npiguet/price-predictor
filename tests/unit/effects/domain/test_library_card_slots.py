"""A library card below its library's top is no slot and trains nothing.

The snapshot holds each library's top card and every library card the record's
own events name (root spec § 8.4): a card below the top is there because it is
drawn, milled or searched for, which is what the head predicts. It gets no
card slot and no per-entity target; its draw, mill or search stays in its
player's counts (root spec § 9.3).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.effect_head_input import SurfaceBuilder, board_entities
from effects.domain.effect_head_layout import LIBRARY_EVENTS, HeadField, LibraryEvent
from effects.domain.effect_targets import derive_targets
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: Wizard Class's level-1 draw: P1 draws E51 (the top) and E50 (rank 1).
_WIZARD_CLASS = f"{_RUN}.2-mwqvwtly1zvq.1629"
#: Deal Gone Bad: P0 mills E30 (the top), E19 (rank 1) and E22 (rank 2).
_DEAL_GONE_BAD = f"{_RUN}.2-mwqvqdxrusnv.313"
#: Elfhame Sanctuary: P1 searches for E61 (rank 1).
_ELFHAME_SANCTUARY = f"{_RUN}.0-mwqvqdyrb9c0.1"
#: Thada Adel: P0 searches P1's library for E78 (rank 20).
_THADA_ADEL = f"{_RUN}.0-mwqvqdyrb9c0.681"

_BELOW_TOP = {
    _WIZARD_CLASS: {"E50"},
    _DEAL_GONE_BAD: {"E19", "E22"},
    _ELFHAME_SANCTUARY: {"E61"},
    _THADA_ADEL: {"E78"},
}


@pytest.fixture(scope="module")
def records():
    wanted = set(_BELOW_TOP)
    return {record.record_id: record
            for record in map(record_from_dict, gen3_fixture_records())
            if record.record_id in wanted}


def _slot_subjects(record) -> set[str]:
    surface = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)
    return {surface.slots[index].subject_id for index in surface.entity_slots}


@pytest.mark.parametrize("record_id", sorted(_BELOW_TOP))
def test_a_library_card_below_the_top_gets_no_slot(records, record_id):
    record = records[record_id]
    below = _BELOW_TOP[record_id]
    assert below <= {entity.id for entity in record.state.entities}
    assert not below & _slot_subjects(record)
    assert not below & {entity.id for entity in board_entities(record)}


@pytest.mark.parametrize("record_id", sorted(_BELOW_TOP))
def test_a_library_card_below_the_top_trains_nothing(records, record_id):
    assert not _BELOW_TOP[record_id] & set(derive_targets(records[record_id]))


def test_the_top_card_keeps_its_slot_and_its_target(records):
    record = records[_WIZARD_CLASS]
    assert "E51" in _slot_subjects(record)
    assert derive_targets(record)["E51"].affected


def test_a_library_card_with_no_rank_gets_no_slot(records):
    """Only rank 0 is known to be the top."""
    record = records[_WIZARD_CLASS]
    unranked = replace(record, state=replace(record.state, entities=tuple(
        replace(entity, order_rank=None) if entity.id == "E51" else entity
        for entity in record.state.entities)))
    assert "E51" not in _slot_subjects(unranked)
    assert "E51" not in derive_targets(unranked)


def test_the_draws_stay_in_the_players_count(records):
    """Wizard Class's half holds no library slot beyond the top card, and
    its player still draws two."""
    targets = derive_targets(records[_WIZARD_CLASS])
    assert targets["P1"].fields[HeadField.CARDS_DRAWN] == 2


def test_the_mills_stay_in_the_players_count(records):
    targets = derive_targets(records[_DEAL_GONE_BAD])
    assert targets["P0"].fields[HeadField.CARDS_MILLED] == 3


def test_a_named_search_stays_in_the_players_library_events(records):
    targets = derive_targets(records[_ELFHAME_SANCTUARY])
    tutor = LIBRARY_EVENTS.index(LibraryEvent.TUTOR)
    assert targets["P1"].fields[HeadField.LIBRARY_EVENTS][tutor] == 1.0


def test_a_search_without_its_named_mode_stays_in_the_players_shuffle(records):
    """Thada Adel's record names no ``SearchedLibrary``: only the shuffle reads."""
    targets = derive_targets(records[_THADA_ADEL])
    events = targets["P1"].fields[HeadField.LIBRARY_EVENTS]
    assert events[LIBRARY_EVENTS.index(LibraryEvent.SHUFFLE)] == 1.0
    assert events[LIBRARY_EVENTS.index(LibraryEvent.TUTOR)] == 0.0
