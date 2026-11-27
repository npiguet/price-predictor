"""T115-T116: relations from the record's links and from the reader's own
layout."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from effects.domain.card_features import card_features, card_segments
from effects.domain.input_layout import (
    ACT_SLOT,
    SlotKind,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import (
    PlayerState,
    Refs,
)
from tests.unit.effects.builders import slots_of_kind
from tests.unit.effects.domain.effect_head_input_support import (
    _ANTHEM,
    _BOLT,
    _CHOICE,
    _RUN,
    _WARD,
    BuildSettings,
    SlotLink,
    _build,
    _gen3_entity,
    _gen3_record,
    _line,
    _linked,
    _relation,
    _subject_slot,
)


class TestAttachedTo:
    """Story 5 acceptance 1 (T115, T116)."""

    def test_an_aura_relates_to_its_creature_and_no_card_input_says_so(self, gen3_records):
        record = gen3_records[f"{_RUN}.0-mwqvqdyrb9c0.240"]
        aura = record.state.entity("E67")
        assert "aura" in aura.subtypes and aura.attached_to == "E43"
        surface = _build(record)
        assert _linked(surface, SlotLink(_subject_slot(surface, "E67"),
                                         _subject_slot(surface, "E43"), "attached_to"))
        assert "attached" not in card_segments(aura, record)
        assert card_features(replace(aura, attached_to=None), record) == card_features(
            aura, record)


def _two_slots(a, b) -> bool:
    """Whether a link's ends ``a`` and ``b`` are two slots, each found."""
    return a is not None and b is not None and a != b


def _board(surface):
    """The triples between cards, players and [ACT], by what their ends
    stand for rather than where they sit."""
    out = set()
    for i, j, r in surface.relations:
        a, b = surface.slots[i], surface.slots[j]
        if SlotKind.ABILITY not in (a.kind, b.kind):
            out.add((a.kind, a.subject_id, b.kind, b.subject_id, r))
    return out


class TestRelations:
    """T116: relations from the record's links and from the reader's own layout."""

    def test_a_cards_lines_its_modes_and_the_lines_of_one_card(self):
        modes = tuple(ProvenanceKey(_CHOICE.script_file, 0, "spell", 0, n) for n in range(2))
        charm = _gen3_entity("E1", lines=(_line(_CHOICE, kind="spell"), _line(_ANTHEM)))
        surface = _build(_gen3_record((charm,)),
                         BuildSettings(options_for=lambda key: modes if key == _CHOICE else ()))
        card = _subject_slot(surface, "E1")
        root, first, second, anthem = card + 1, card + 2, card + 3, card + 4
        for line in (root, first, second, anthem):
            assert _linked(surface, SlotLink(line, card, "line_of"))
        assert _linked(surface, SlotLink(first, root, "mode_of"))
        assert _linked(surface, SlotLink(second, root, "mode_of"))
        assert not _linked(surface, SlotLink(anthem, root, "mode_of"))
        same = [t for t in surface.relations if t[2] in (
            _relation("line_of_the_same_card"), _relation("line_of_the_same_card", "reverse"))]
        assert len(same) == 4 * 3

    def test_same_names_and_a_chosen_name_relate_and_a_face_down_card_does_not(self):
        from effects.domain.state_snapshot import CardChoices

        record = _gen3_record((
            _gen3_entity("E1", name="Plains"), _gen3_entity("E2", name="plains"),
            _gen3_entity("E3", name="Island"),
            _gen3_entity("E4", name="Pithing Needle", chosen=CardChoices(names=("Plains",))),
            _gen3_entity("E5", name="Plains", face_state="face_down"),
        ))
        surface = _build(record)
        e1, e2, e3, e4, e5 = (_subject_slot(surface, f"E{n}") for n in range(1, 6))
        assert _linked(surface, SlotLink(e1, e2, "has_the_same_name"))
        assert _linked(surface, SlotLink(e4, e1, "has_the_same_name"))
        assert _linked(surface, SlotLink(e4, e2, "has_the_same_name"))
        named = {end for t in surface.relations if t[2] in (
            _relation("has_the_same_name"), _relation("has_the_same_name", "reverse"))
            for end in t[:2]}
        assert named == {e1, e2, e4}
        assert e3 not in named and e5 not in named

    def test_same_names_relate_on_the_battlefield_the_stack_and_in_hand_only(self):
        """A pair relates when both cards are in those zones; a copy counts by
        the zone it last had; a chooser relates to its name in every zone."""
        from effects.domain.state_snapshot import CardChoices

        record = _gen3_record((
            _gen3_entity("E1", name="Plains", zone="battlefield"),
            _gen3_entity("E2", name="Plains", zone="hand"),
            _gen3_entity("E3", name="Island", zone="graveyard"),
            _gen3_entity("E4", name="Island", zone="graveyard"),
            _gen3_entity("E5", name="Swamp", zone="graveyard"),
            _gen3_entity("E6", name="Swamp", zone="stack"),
            _gen3_entity("L7", name="Forest", zone="battlefield", role="last_known"),
            _gen3_entity("E8", name="Forest", zone="hand"),
            _gen3_entity("E9", name="Cranial Extraction", zone="stack",
                         chosen=CardChoices(names=("Island",))),
        ))
        surface = _build(record)
        e = {n: _subject_slot(surface, n) for n in
             ("E1", "E2", "E3", "E4", "E5", "E6", "L7", "E8", "E9")}
        assert _linked(surface, SlotLink(e["E1"], e["E2"], "has_the_same_name"))
        assert not _linked(surface, SlotLink(e["E3"], e["E4"], "has_the_same_name"))
        assert not _linked(surface, SlotLink(e["E5"], e["E6"], "has_the_same_name"))
        assert _linked(surface, SlotLink(e["L7"], e["E8"], "has_the_same_name"))
        assert _linked(surface, SlotLink(e["E9"], e["E3"], "has_the_same_name"))
        assert _linked(surface, SlotLink(e["E9"], e["E4"], "has_the_same_name"))

    def test_the_act_slot_targets_by_index_and_reads_its_source(self):
        record = _gen3_record(
            (_gen3_entity("E1"), _gen3_entity("E2")),
            refs=Refs(targets=("E2", "P1"), source="E1"),
        )
        surface = _build(record)
        assert _linked(surface, SlotLink(ACT_SLOT, _subject_slot(surface, "E2"), "targets", 0))
        assert _linked(surface, SlotLink(ACT_SLOT, _subject_slot(surface, "P1"), "targets", 1))
        assert _linked(surface, SlotLink(_subject_slot(surface, "E1"), ACT_SLOT, "source_of"))

    def test_a_target_index_past_the_last_type_shares_it(self):
        from effects.domain.relation_types import TARGET_INDICES
        from effects.domain.state_snapshot import LineRef, Link

        record = _gen3_record(
            (_gen3_entity("E1", lines=(_line(_BOLT, kind="spell"),)), _gen3_entity("E2")),
            links=(Link(LineRef("E1", _BOLT), "E2", "targets", 5),))
        surface = _build(record)
        line = _subject_slot(surface, "E1") + 1
        assert _linked(surface, SlotLink(line, _subject_slot(surface, "E2"), "targets",
                                         TARGET_INDICES - 1))

    @pytest.mark.parametrize("why", ["no line", "stripped"])
    def test_a_link_from_a_line_with_no_slot_reads_its_card(self, why):
        from effects.domain.state_snapshot import LineRef, Link

        ward = _line(_WARD, kind="keyword", fragment="Ward:2")
        record = _gen3_record(
            (_gen3_entity("E1", lines=(ward, _line(_ANTHEM))), _gen3_entity("E2")),
            links=(Link(LineRef("E1", _WARD), "E2", "affects"),))
        if why == "no line":
            surface = _build(record, BuildSettings(has_line=lambda key: key != _WARD))
        else:
            surface = _build(record, BuildSettings(stripped={"E1": frozenset({"ward"})}))
        card = _subject_slot(surface, "E1")
        assert [slot.entity_id for slot in slots_of_kind(surface, SlotKind.ABILITY)] == ["E1"]
        assert _linked(surface, SlotLink(card, _subject_slot(surface, "E2"), "affects"))
        assert _linked(surface, SlotLink(card + 1, card, "line_of"))

    def test_exploration_affects_its_controllers_player_slot(self):
        """Story 5 acceptance 8."""
        from effects.domain.input_vocabularies import THIS_TURN_COUNTERS
        from effects.domain.state_snapshot import LineRef, Link

        exploration_static = ProvenanceKey("cardsfolder/e/exploration.txt", 0, "static", 0)
        exploration = _gen3_entity("E1", name="Exploration", types=("enchantment",), pt=None,
                                   lines=(_line(exploration_static, affected=("P0",)),))
        players = (
            PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0,
                        this_turn={"lands_played": 1}, max_land_plays=2),
            PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0, max_land_plays=1),
        )
        record = _gen3_record(
            (exploration,), players=players,
            links=(Link(LineRef("E1", exploration_static), "P0", "affects"),))
        surface = _build(record)
        player = _subject_slot(surface, "P0")
        lands = 6 * 2 + THIS_TURN_COUNTERS.index("lands_played") * 2
        limit = 6 * 2 + len(THIS_TURN_COUNTERS) * 2
        features = surface.slots[player].features
        assert features[lands] == 1.0 and features[limit] == 2.0
        static = _subject_slot(surface, "E1") + 1
        assert _linked(surface, SlotLink(static, player, "affects"))
        assert not _linked(surface, SlotLink(static, _subject_slot(surface, "P1"), "affects"))

    def test_every_fixture_link_between_two_slots_is_two_triples(self, gen3_records):
        checked = 0
        for record in gen3_records.values():
            surface = _build(record)
            subjects = {slot.subject_id: index for index, slot in enumerate(surface.slots)
                        if slot.kind in (SlotKind.CARD, SlotKind.PLAYER)}
            for found in record.state.links:
                a, b = subjects.get(found.from_), subjects.get(found.to)
                if not _two_slots(a, b):
                    continue
                index = (found.index or 0) if found.type == "targets" else None
                assert _linked(surface, SlotLink(a, b, found.type, index)), (
                    record.record_id, found)
                checked += 1
        assert checked > 10_000

    def test_relations_survive_context_dropout_with_remapped_indices(self, gen3_records):
        record = max(gen3_records.values(),
                     key=lambda r: sum(1 for link in r.state.links
                                       if not isinstance(link.from_, str)))
        full = _build(record)
        dropped = _build(record, BuildSettings(rng=random.Random(3), context_dropout=0.5))
        assert len(slots_of_kind(dropped, SlotKind.ABILITY)) < len(
            slots_of_kind(full, SlotKind.ABILITY))
        assert all(max(i, j) < len(dropped.slots) for i, j, _ in dropped.relations)

        line_of = _relation("line_of")
        for i, j, r in dropped.relations:
            if r == line_of:
                assert dropped.slots[i].kind is SlotKind.ABILITY
                assert dropped.slots[j].kind is SlotKind.CARD
                assert dropped.slots[i].entity_id == dropped.slots[j].entity_id

        # A dropped line's links read its card: only ever more card-level triples.
        assert _board(full) <= _board(dropped)

    def test_two_face_down_cards_build_the_same_relations(self):
        from effects.domain.state_snapshot import RealFace

        def board(real_name):
            hidden = _gen3_entity(
                "E1", name="", face_state="face_down", types=("creature",),
                real_face=RealFace(name=real_name), lines=(_line(_ANTHEM, active=False),))
            return _gen3_record((hidden, _gen3_entity("E2", name=real_name)))

        assert _build(board("Serra Angel")).relations == _build(board("Grizzly Bears")).relations
