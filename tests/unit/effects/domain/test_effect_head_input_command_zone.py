"""Root spec § 9.3 (T114): floating effects, delayed triggers and the
other extra cards are card slots their links relate."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from effects.domain.input_layout import SlotKind
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import GlobalState
from tests.unit.effects.builders import slots_of_kind
from tests.unit.effects.domain.effect_head_input_support import (
    _RUN,
    _build,
    _card_part,
    _flags,
    _gen3_entity,
    _gen3_record,
    _line,
    _linked,
    _subject_slot,
)


def _with_mutate_component(record, component_id, creature_id):
    """``record`` with a component merged under its battlefield creature, as
    the collector writes one (role ``mutate_component``, zone ``merged``,
    linked ``merged_with``): the fixture's pilot mutated nothing."""
    from effects.domain.state_snapshot import Link

    creature = record.state.entity(creature_id)
    assert creature.zone == "battlefield" and "creature" in creature.types
    component = replace(creature, id=component_id, zone="merged", role="mutate_component")
    return replace(record, state=replace(
        record.state, entities=(*record.state.entities, component),
        links=(*record.state.links, Link(from_=component_id, to=creature_id,
                                         type="merged_with"))))


_FALTER = ProvenanceKey("cardsfolder/f/falter.txt", 0, "spell", 0)


class TestCommandZoneSlots:
    """Root spec § 9.3 (T114) and story 5 acceptance 2."""

    def _falter_record(self, **effect):
        from effects.domain.state_snapshot import CommandZoneObject, LineRef, Link

        floating = CommandZoneObject(
            id="F40", source_lines=(_FALTER,), affected=("E1", "E2"),
            duration=effect.pop("duration", "end_of_turn"), **effect)
        caster = _gen3_entity("E3", name="Falter", zone="graveyard",
                              lines=(_line(_FALTER, kind="spell"),))
        return _gen3_record(
            (_gen3_entity("E1"), _gen3_entity("E2", controller="P1"), caster),
            links=(Link("F40", "E1", "affects"), Link("F40", "E2", "affects"),
                   Link("F40", LineRef("E3", _FALTER), "created_by")),
            global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0",
                                stack_size=0, floating_effects=(floating,)),
        )

    def test_creatures_cant_block_this_turn_is_a_command_zone_card(self):
        from effects.domain.input_vocabularies import EFFECT_DURATIONS
        from effects.domain.surface_builder import command_zone_entities

        record = self._falter_record()
        falter_e = (0.5, 0.5, 0.0, 0.0)
        surface = _build(record, e_for=lambda key: falter_e if key == _FALTER else None)
        at = _subject_slot(surface, "F40")
        assert at not in surface.entity_slots
        line = surface.slots[at + 1]
        assert line.kind is SlotKind.ABILITY and line.entity_id == "F40"
        assert line.e_vector == falter_e and _flags(line)["granted"] == 1.0
        # Its lines are an effect's, whose duration the card reads apart.
        assert _flags(line)["by_effect"] == 1.0
        for creature in ("E1", "E2"):
            assert _linked(surface, at, _subject_slot(surface, creature), "affects")
        # created_by ends at the casting line's own slot, not its card.
        caster_line = _subject_slot(surface, "E3") + 1
        assert _linked(surface, at, caster_line, "created_by")
        effect, = command_zone_entities(record)
        assert effect.controller == "P0" and effect.zone == "command"
        duration = _card_part(effect, record, "command_zone")
        assert duration[EFFECT_DURATIONS.index("end_of_turn")] == 1.0
        assert sum(duration[:len(EFFECT_DURATIONS) + 1]) == 1.0

    def test_a_shield_reads_what_it_has_left(self):
        from effects.domain.input_vocabularies import EFFECT_DURATIONS
        from effects.domain.surface_builder import command_zone_entities

        record = self._falter_record(duration="until_leaves", shield_left=3)
        effect, = command_zone_entities(record)
        part = _card_part(effect, record, "command_zone")
        assert part[EFFECT_DURATIONS.index("until_leaves")] == 1.0
        assert part[len(EFFECT_DURATIONS) + 1:] == pytest.approx((1.0, 3.0, math.log1p(3)))
        # Any other card reads zeros there.
        assert sum(_card_part(record.state.entities[0], record, "command_zone")) == 0.0

    def test_an_effect_card_on_the_board_reads_its_duration_and_gets_no_second_slot(self):
        from effects.domain.input_vocabularies import EFFECT_DURATIONS
        from effects.domain.state_snapshot import CommandZoneObject

        card = _gen3_entity("E81", name="Regeneration", zone="command", lines=(
            _line(_FALTER, kind="trigger", granted=True),))
        record = _gen3_record((card,), global_=GlobalState(
            turn=3, phase="main1", active="P0", priority="P0", stack_size=0,
            floating_effects=(CommandZoneObject(id="E81", source_lines=(_FALTER,)),)))
        surface = _build(record)
        assert len(slots_of_kind(surface, SlotKind.CARD)) == 1
        assert _card_part(card, record, "command_zone")[
            EFFECT_DURATIONS.index("permanent")] == 1.0

    def test_a_real_floating_effect_affects_each_creature_it_pumps(self, gen3_records):
        record = gen3_records[f"{_RUN}.0-mwqvqdyrb9c0.81"]
        floating = next(f for f in record.state.global_.floating_effects if f.id == "F37")
        source = floating.source_lines[0]
        vector = (0.0, 0.0, 0.0, 1.0)
        surface = _build(record, e_for=lambda key: vector if key == source else None)
        at = _subject_slot(surface, "F37")
        assert surface.slots[at + 1].e_vector == vector
        assert floating.affected == ("E52", "E173", "E174")
        for creature in floating.affected:
            assert _linked(surface, at, _subject_slot(surface, creature), "affects")

    def test_a_real_delayed_trigger_is_a_card_reading_its_source_line(self, gen3_records):
        record = gen3_records[f"{_RUN}.0-mwqvqdyrb9c0.424"]
        trigger, = record.state.global_.delayed_triggers
        vector = (0.0, 1.0, 1.0, 0.0)
        surface = _build(record, e_for=lambda key: vector if key in trigger.source_lines else None)
        at = _subject_slot(surface, trigger.id)
        assert surface.slots[at + 1].e_vector == vector
        # Pending on a card of the board: the record's own link relates them.
        assert trigger.remembered == ("E240",)
        assert _linked(surface, at, _subject_slot(surface, "E240"), "pending_on")

    def test_a_delayed_trigger_is_pending_on_its_card(self):
        from effects.domain.state_snapshot import CommandZoneObject, Link

        record = _gen3_record((_gen3_entity("E1"),), links=(Link("D7", "E1", "pending_on"),),
                              global_=GlobalState(
                                  turn=3, phase="main1", active="P0", priority="P0",
                                  stack_size=0, delayed_triggers=(CommandZoneObject(
                                      id="D7", source_lines=(_FALTER,), remembered=("E1",)),)))
        surface = _build(record)
        assert _linked(surface, _subject_slot(surface, "D7"), _subject_slot(surface, "E1"),
                       "pending_on")

    @pytest.mark.parametrize(("record_id", "family", "a", "b"), [
        (f"{_RUN}.0-mwqvqdyrb9c0.5", "last_known_state_of", "L74", "E74"),
        (f"{_RUN}.0-mwqvqdyrb9c0.81", "merged_with", "E9001", "E52"),
        (f"{_RUN}.2-mwqvwtly1zvq.1597", "in_dungeon", "E81", "P0"),
    ])
    def test_the_extra_cards_are_slots_their_links_relate(
        self, gen3_records, record_id, family, a, b,
    ):
        record = gen3_records[record_id]
        if family == "merged_with":
            record = _with_mutate_component(record, a, b)
        surface = _build(record)
        assert _linked(surface, _subject_slot(surface, a), _subject_slot(surface, b), family)

    def test_a_librarys_top_card_reads_order_rank_zero(self, gen3_records):
        """A card below the top gets no slot (``test_library_card_slots.py``)."""
        record, entity = next(
            (record, entity) for record in gen3_records.values()
            for entity in record.state.entities
            if entity.role == "library_card" and entity.order_rank == 0)
        assert _card_part(entity, record, "order_rank")[:2] == (1.0, 0.0)
        surface = _build(record)
        assert _subject_slot(surface, entity.id) in surface.entity_slots
