"""A card slot says what kind of entity it is (findings 4.7, B10).

A gen-3 snapshot holds entities beyond the four tiers, each with a ``role``
(``state_snapshot.EntityRole``). A last-known copy keeps the zone it last had,
so without its role it reads exactly as the live card there: an ``L<id>``
copy with zone ``battlefield`` read as a permanent on the battlefield. The
card slot carries the role as a one-hot, reads summoning sickness on a
permanent only, and gives a last-known copy no same-name relation with its
live twin, which it merely is.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.input_layout import SlotKind
from effects.domain.input_vocabularies import ENTITY_ROLES, ZONES
from effects.domain.relation_types import relation_type
from effects.domain.surface_builder import SurfaceBuilder
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: Aven Heartstabber's dies trigger: L7, its last-known copy, has zone
#: ``battlefield`` and is summoning sick; E7, the card, is in the graveyard.
_HEARTSTABBER = f"{_RUN}.2-mwqvqdxrusnv.783"
#: A Clue sacrificed for its own ability: E86 ceased to exist and is written
#: as it last existed, role ``last_known``, zone null.
_CEASED_CLUE = f"{_RUN}.2-mwqvqdxrusnv.7045"


@pytest.fixture(scope="module")
def records():
    wanted = {_HEARTSTABBER, _CEASED_CLUE}
    return {record.record_id: record
            for record in map(record_from_dict, gen3_fixture_records())
            if record.record_id in wanted}


def _role_columns(entity, record) -> tuple[float, ...]:
    return card_features(entity, record)[card_segments(entity, record)["role"]]


def _one_hot(role: str) -> tuple[float, ...]:
    return tuple(1.0 if member == role else 0.0 for member in (*ENTITY_ROLES, "other"))


def _sick_column(entity, record) -> float:
    status = card_features(entity, record)[card_segments(entity, record)["status"]]
    # the zone one-hot with its "other" slot, then tapped, then sick
    return status[len(ZONES) + 2]


def test_a_last_known_copy_on_the_battlefield_is_not_a_permanent(records):
    record = records[_HEARTSTABBER]
    copy = record.state.entity("L7")
    assert (copy.zone, copy.role) == ("battlefield", "last_known")
    permanent = replace(copy, role=None)
    assert card_features(copy, record) != card_features(permanent, record)
    assert _role_columns(copy, record) == _one_hot("last_known")
    assert _role_columns(permanent, record) == _one_hot("tier")


def test_every_role_reads_its_own_column(records):
    record = records[_HEARTSTABBER]
    entity = record.state.entity("E7")
    for role in ENTITY_ROLES[1:]:
        assert _role_columns(replace(entity, role=role), record) == _one_hot(role)


def test_a_token_that_ceased_to_exist_reads_last_known(records):
    """The collector writes a ceased token zoneless (record-format.md); its
    slot reads no zone and the last-known role."""
    record = records[_CEASED_CLUE]
    clue = record.state.entity("E86")
    assert (clue.zone, clue.role) == (None, "last_known")
    assert _role_columns(clue, record) == _one_hot("last_known")


def test_summoning_sickness_reads_on_a_permanent_only(records):
    record = records[_HEARTSTABBER]
    copy = record.state.entity("L7")
    assert copy.sick
    assert _sick_column(copy, record) == 0.0
    assert _sick_column(replace(copy, role=None), record) == 1.0
    for zone in ("hand", "graveyard", "library", "exile", "stack"):
        assert _sick_column(replace(copy, role=None, zone=zone), record) == 0.0


def _card_slot(surface, entity_id: str) -> int:
    return next(index for index, slot in enumerate(surface.slots)
                if slot.kind is SlotKind.CARD and slot.entity_id == entity_id)


def test_a_last_known_copy_has_no_same_name_relation_with_its_card(records):
    record = records[_HEARTSTABBER]
    surface = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)
    copy, card = _card_slot(surface, "L7"), _card_slot(surface, "E7")
    same_name = relation_type("has_the_same_name")
    assert not {(copy, card, same_name), (card, copy, same_name)} & set(surface.relations)
    assert (copy, card, relation_type("last_known_state_of")) in surface.relations


def test_two_cards_of_one_name_keep_their_relation(records):
    """Only the copy and its own card lose it: the copy, which last had the
    battlefield, and a namesake in a hand are still related. The card itself
    is in the graveyard now, so it relates to no namesake in a hand."""
    record = records[_HEARTSTABBER]
    card = record.state.entity("E7")
    namesake = replace(card, id="E9999", zone="hand")
    board = replace(record, state=replace(
        record.state, entities=(*record.state.entities, namesake)))
    surface = SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(board)
    same_name = relation_type("has_the_same_name")
    assert (_card_slot(surface, "E7"), _card_slot(surface, "E9999"), same_name) \n        not in surface.relations
    assert (_card_slot(surface, "L7"), _card_slot(surface, "E9999"), same_name) \n        in surface.relations
