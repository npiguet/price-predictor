"""What the targets read of the gen-3 wire's second and third batches of members."""

from __future__ import annotations

import pytest

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    PlayabilitySubkind,
    RecordKind,
    TriggerPayload,
)
from tests.unit.effects.domain.effect_targets_support import (
    Board,
    _entity,
    _on_board,
    _record,
    _resolution,
    _zone,
)


class TestTheWp4Wire:
    """What the targets read of the gen-3 wire's second and third batches of members
    (``record-schema-delta.md``): each decision is stated in
    ``src/effects/docs/training.md`` § Losses."""

    def test_a_hidden_keyword_change_sets_no_keyword_column_and_is_counted(self):
        """Forge's text (``CARDNAME can't block.``) names no keyword column: the
        entity is affected, its keyword fields keep their silent value, and the
        record counts the change as not supervised."""
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["CARDNAME can't block."], "removed": False,
                          "hidden": True}),
        ))
        assert targets["E1"].affected
        assert "keywords_gained" not in targets["E1"].fields
        assert targets.unsupervised == {"hidden_keyword_change": 1}

    @pytest.mark.parametrize(("event_type", "params"), [
        (EventType.NAME_CHANGE, {"name": "Lord of Tresserhorn"}),
        (EventType.MANA_COST_CHANGE, {"mana_cost": {"generic": 2}, "mana_value": 2}),
    ])
    def test_a_name_or_mana_cost_change_has_no_column_and_is_counted(self, event_type, params):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=("E1",), params=params)))
        assert targets["E1"].affected and targets["E1"].fields == {}
        assert targets.unsupervised == {event_type.value: 1}

    def test_wither_damage_is_no_marked_damage(self):
        """The -1/-1 counters arrive as the record's own ``counter_change``."""
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2, "combat": True, "damage_type": "m1m1_counters"}),
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "M1M1", "delta": 2}),
        ))
        fields = targets["E1"].fields
        assert "damage_taken" not in fields and fields["counters_delta_m1m1"] == 2.0

    def test_a_deathtouch_witherer_s_damage_is_counted_once(self):
        """Forge reads ``deathtouch`` for a deathtouch source with wither: the
        hit's counters arrive as a ``counter_change`` of its amount, so the
        damage is not marked as well."""
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3, "combat": True, "damage_type": "deathtouch"}),
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "M1M1", "delta": 3}),
            Event(type=EventType.DAMAGE_DEALT, subjects=("E2",),
                  params={"amount": 1, "combat": True, "damage_type": "deathtouch"}),
        ))
        assert targets["E1"].fields.get("damage_taken", 0) == 0
        assert targets["E2"].fields["damage_taken"] == 1.0

    def test_a_destruction_a_replacement_turns_into_exile_reads_as_exiled(self):
        """The bus writes ``destroyed`` just before the permanent's real move."""
        targets = derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "battlefield", "to_zone": "exile"}),
        ))
        assert targets["E1"].fields["zone_outcome"] == _zone("exiled")

    def test_a_fizzled_spell_s_card_reads_its_real_move(self):
        """Its half holds exactly its card's move off the stack; a copy's holds none."""
        fizzled = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "stack", "to_zone": "graveyard"}),
        ))
        assert fizzled["E1"].affected
        assert fizzled["E1"].fields["zone_outcome"] == _zone("died")
        assert derive_targets(_resolution()) == {}

    def test_a_batch_move_places_each_card(self):
        """A ``ChangesZoneAll`` batch names several subjects; a resolution's
        move of several cards places each, a trigger's event none."""
        move = Event(type=EventType.ZONE_CHANGE, subjects=("E1", "E2"),
                     params={"from_zone": "battlefield", "to_zone": "exile"})
        targets = derive_targets(_resolution(move))
        assert {s: t.fields["zone_outcome"] for s, t in targets.items()} == {
            "E1": _zone("exiled"), "E2": _zone("exiled")}
        trigger = _record(TriggerPayload(event=move, fired=True), kind=RecordKind.TRIGGER,
                          moment=None)
        assert all(not t.affected for t in derive_targets(trigger).values())

    def test_a_face_change_names_the_face_now_up(self):
        from effects.domain.input_vocabularies import FACE_STATES

        targets = derive_targets(_resolution(
            Event(type=EventType.FACE_CHANGE, subjects=("E1",), params={"to_state": "back"})))
        assert targets["E1"].fields["face_after"] == FACE_STATES.index("back")

    def test_a_phased_out_planeswalker_is_no_attack_target_at_all(self):
        """A phased-out permanent is an entity of the board, treated as though
        it does not exist (702.26b): no defender to be legal or not."""
        from effects.domain.records import PlayabilityAttackersPayload

        walker = _entity("E9", types=("planeswalker",), controller="P1")
        gone = _entity("E8", types=("planeswalker",), controller="P1",
                       designations=("phased_out",))
        assert walker.on_battlefield and not gone.on_battlefield
        record = _on_board(PlayabilityAttackersPayload(
            legal_attackers=("E1",), anchor_attacker="E1", legal_defenders=("P1", "E9")),
            Board.of(_entity("E1"), walker, gone), kind=RecordKind.PLAYABILITY,
            subkind=PlayabilitySubkind.ATTACKERS)
        targets = derive_targets(record)
        assert targets["E9"].fields["attack_target_legal"] == 1
        assert "E8" not in targets
