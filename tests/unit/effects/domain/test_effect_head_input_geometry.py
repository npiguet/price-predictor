"""The effect head's token geometry (T043): the surface opens with
``[GLOBAL]`` then ``[ACT]``, players before cards; position ids restart at
each ``[CARD]``; ``[ACT]`` is empty where no single line acts; every slot
of one kind has one width."""

from __future__ import annotations

from effects.domain.card_features import card_features
from effects.domain.input_layout import (
    ACT_SLOT,
    SlotKind,
    act_features,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
)
from tests.unit.effects.builders import (
    lines_of,
    slots_of_kind,
)
from tests.unit.effects.domain.effect_head_input_support import (
    _ANTHEM,
    _BOLT,
    _VECTORS,
    _ZERO,
    _build,
    _entity,
    _record,
    _snapshot,
)


class TestSurfaceShape:
    def test_the_surface_opens_with_global_then_act(self):
        surface = _build(_record())
        assert surface.slots[0].kind is SlotKind.GLOBAL
        assert surface.slots[1].kind is SlotKind.ACT
        assert ACT_SLOT == 1

    def test_players_come_before_cards(self):
        kinds = [slot.kind for slot in _build(_record()).slots]
        assert kinds.index(SlotKind.PLAYER) < kinds.index(SlotKind.CARD)

    def test_every_player_and_entity_gets_a_slot(self):
        surface = _build(_record())
        assert len(slots_of_kind(surface, SlotKind.PLAYER)) == 2
        assert len(slots_of_kind(surface, SlotKind.CARD)) == 2

    def test_the_per_entity_head_targets_every_card_and_player(self):
        """FR-081: the gate trains on every entity, players included."""
        surface = _build(_record())
        targeted = {surface.slots[i].kind for i in surface.entity_slots}
        assert targeted == {SlotKind.PLAYER, SlotKind.CARD}
        assert len(surface.entity_slots) == 4

    def test_an_entitys_abilities_follow_its_card_token(self):
        surface = _build(_record())
        kinds = [slot.kind for slot in surface.slots]
        first_card = kinds.index(SlotKind.CARD)
        assert kinds[first_card + 1] is SlotKind.ABILITY
        assert surface.slots[first_card + 1].entity_id == "E1"



class TestPositionReset:
    def test_positions_run_up_to_the_first_card(self):
        surface = _build(_record())
        leading = [
            slot.position for slot in surface.slots
            if slot.kind in (SlotKind.GLOBAL, SlotKind.ACT, SlotKind.PLAYER)
        ]
        assert leading == [0, 1, 2, 3]

    def test_each_card_restarts_at_zero(self):
        surface = _build(_record())
        for slot in slots_of_kind(surface, SlotKind.CARD):
            assert slot.position == 0

    def test_an_entitys_abilities_number_upward_from_its_card(self):
        surface = _build(_record())
        positions = [
            (slot.kind, slot.position) for slot in surface.slots
            if slot.kind in (SlotKind.CARD, SlotKind.ABILITY)
        ]
        assert positions == [
            (SlotKind.CARD, 0), (SlotKind.ABILITY, 1),
            (SlotKind.CARD, 0), (SlotKind.ABILITY, 1),
        ]

    def test_a_board_position_does_not_change_an_entitys_geometry(self):
        """The reset is what makes twelve permanents one geometry, not twelve."""
        one = _snapshot(entities=(_entity("E1", lines=lines_of(printed=(_ANTHEM,))),))
        many = _snapshot(entities=(
            _entity("E0"), _entity("E1", lines=lines_of(printed=(_ANTHEM,))),
        ))
        first = _build(_record(state=one))
        second = _build(_record(state=many))
        target_first = [s for s in first.slots if s.entity_id == "E1"]
        target_second = [s for s in second.slots if s.entity_id == "E1"]
        assert [s.position for s in target_first] == [
            s.position for s in target_second
        ]


class TestActSlot:
    def test_a_resolution_carries_its_acting_lines_vector(self):
        assert _build(_record()).slots[1].e_vector == _VECTORS[_BOLT]

    def test_act_is_empty_for_a_combat_record(self):
        surface = _build(_record(kind=RecordKind.COMBAT, ability=None))
        assert surface.slots[1].e_vector == _ZERO
        assert set(surface.slots[1].features) == {0.0}

    def test_act_is_empty_for_the_attackers_subkind(self):
        from effects.domain.records import PlayabilityAttackersPayload

        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.ATTACKERS,
            payload=PlayabilityAttackersPayload(),
        )
        assert set(_build(record).slots[1].features) == {0.0}

    def test_act_is_empty_for_the_blockers_subkind(self):
        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.BLOCKERS,
            payload=PlayabilityBlockersPayload(anchor_attacker="E1"),
        )
        assert set(_build(record).slots[1].features) == {0.0}

    def test_a_populated_act_slot_says_so_in_its_first_feature(self):
        assert act_features(_record())[0] == 1.0
        assert act_features(_record(kind=RecordKind.COMBAT, ability=None))[0] == 0.0

    def test_a_decision_record_puts_its_candidates_vector_in_act(self):
        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.DECISION,
            payload=PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(_BOLT,), can_play=True, affordable=True,
                          has_legal_target=True),
                Candidate(ability=(_ANTHEM,), can_play=False, affordable=False,
                          has_legal_target=False),
            )),
        )
        assert _build(record).slots[1].e_vector == _VECTORS[_BOLT]

    def test_an_unresolvable_key_contributes_zeros_rather_than_dropping(self):
        missing = ProvenanceKey("cardsfolder/x/absent.txt", 0, "spell", 0)
        assert _build(_record(ability=(missing,))).slots[1].e_vector == _ZERO


class TestFeatureWidths:
    def test_every_card_slot_has_the_same_width(self):
        surface = _build(_record())
        widths = {len(slot.features) for slot in slots_of_kind(surface, SlotKind.CARD)}
        assert len(widths) == 1

    def test_every_player_slot_has_the_same_width(self):
        surface = _build(_record())
        widths = {len(slot.features) for slot in slots_of_kind(surface, SlotKind.PLAYER)}
        assert len(widths) == 1

    def test_act_width_is_the_same_whether_populated_or_empty(self):
        populated = len(act_features(_record()))
        empty = len(act_features(_record(kind=RecordKind.COMBAT, ability=None)))
        assert populated == empty

    def test_an_entity_without_power_toughness_keeps_the_width(self):
        creature = _entity("E1")
        land = _entity("E2", pt=None, types=("Land",))
        record = _record(state=_snapshot(entities=(creature, land)))
        assert len(card_features(creature, record)) == len(
            card_features(land, record)
        )
