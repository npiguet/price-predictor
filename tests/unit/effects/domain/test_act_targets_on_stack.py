"""What the acting item targets on the stack: a spell, or an ability (finding 3.3, B12).

``refs.targets_on_stack`` names a target on the stack by its card, a spell
and an ability alike. The acting item's ``targets`` link tells them apart: it
ends at the card for a spell, at the line for an ability (pilot records:
Interdict on Rootwater Hunter's ability ends at the ability's line, Make
Disappear on a spell that has since left the stack ends at the card). A
permanent whose ability is targeted is not itself targeted, so ``[ACT]``
reads no ``targets`` relation to its card and its targeted bit is clear; the
record's own link still relates the two lines. A spell that has left the
stack since it was targeted still reads targeted.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.input_layout import ACT_SLOT, SlotKind
from effects.domain.relation_types import relation_type
from effects.domain.state_snapshot import LineRef, Link
from effects.domain.surface_builder import SurfaceBuilder
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.domain.test_input_outcome_independence import E_DIM, _Lines

#: Spell Contortion's effect half: E10 targets Searing Blaze, E64, on the stack.
_CONTORTION = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.406"


@pytest.fixture(scope="module")
def contortion():
    return next(record for record in map(record_from_dict, gen3_fixture_records())
                if record.record_id == _CONTORTION)


def _surface(record):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


def _card_slot(surface, entity_id: str) -> int:
    return next(index for index, slot in enumerate(surface.slots)
                if slot.kind is SlotKind.CARD and slot.entity_id == entity_id)


def _act_targets(record, entity_id: str) -> bool:
    surface = _surface(record)
    return (ACT_SLOT, _card_slot(surface, entity_id),
            relation_type("targets", target_index=0)) in surface.relations


def _targeted_bit(record, entity_id: str) -> float:
    entity = record.state.entity(entity_id)
    # the relation-to-act part: controlled by the actor, then targeted
    return card_features(entity, record)[card_segments(entity, record)["relation_to_act"]][1]


def _moved(record, entity_id: str, zone: str):
    return replace(record, state=replace(record.state, entities=tuple(
        replace(entity, zone=zone) if entity.id == entity_id else entity
        for entity in record.state.entities)))


def _aimed_at_an_ability(record):
    """The acting item retargeted at the first line of a battlefield
    permanent, written as the collector writes an ability target."""
    host = next(entity for entity in record.state.entities
                if entity.zone == "battlefield" and entity.lines and entity.lines[0].keys)
    line = LineRef(entity=host.id, key=host.lines[0].keys[0])
    acting = next(link for link in record.state.links
                  if link.type == "targets" and link.to == "E64")
    links = tuple(Link(from_=acting.from_, to=line, type="targets", index=0)
                  if link is acting else link for link in record.state.links)
    refs = replace(record.state.refs, targets=(host.id,), targets_at_cast=(host.id,),
                   targets_on_stack=(host.id,))
    return host.id, replace(record, state=replace(record.state, refs=refs, links=links))


def test_a_spell_on_the_stack_is_targeted(contortion):
    assert _act_targets(contortion, "E64")
    assert _targeted_bit(contortion, "E64") == 1.0


def test_a_spell_that_has_left_the_stack_is_still_targeted(contortion):
    gone = _moved(contortion, "E64", "graveyard")
    assert _act_targets(gone, "E64")
    assert _targeted_bit(gone, "E64") == 1.0


def test_a_permanent_whose_ability_is_targeted_is_not_targeted(contortion):
    host, record = _aimed_at_an_ability(contortion)
    assert not _act_targets(record, host)
    assert _targeted_bit(record, host) == 0.0


def test_the_record_s_link_still_relates_the_two_lines(contortion):
    host, record = _aimed_at_an_ability(contortion)
    surface = _surface(record)
    host_lines = {index for index, slot in enumerate(surface.slots)
                  if slot.kind is SlotKind.ABILITY and slot.entity_id == host}
    targets = relation_type("targets", target_index=0)
    assert any(b in host_lines for _, b, kind in surface.relations if kind == targets)


def test_an_unlinked_target_on_the_stack_reads_its_zone(contortion):
    """With no link from the acting item, a card on the stack is the spell
    and one elsewhere is an ability's host."""
    unlinked = replace(contortion, state=replace(contortion.state, links=()))
    assert _targeted_bit(unlinked, "E64") == 1.0
    assert _targeted_bit(_moved(unlinked, "E64", "battlefield"), "E64") == 0.0
