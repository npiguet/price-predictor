"""The numeric fields events write: damage, life, counters, power and
toughness, mana and poison, signed by the event's mode, and every simple
event type pinned field by field."""

from __future__ import annotations

import pytest

from effects.domain.effect_head_layout import LIBRARY_EVENTS, ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    RecordKind,
    RewritePayload,
)
from tests.unit.effects.domain.effect_targets_support import (
    _record,
    _resolution,
)


class TestMagnitudes:
    def test_damage_accumulates_across_events(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert targets["E1"].fields["damage_taken"] == 5.0

    def test_healing_subtracts_from_damage(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 4}),
            Event(type=EventType.DAMAGE_HEALED, subjects=("E1",),
                  params={"amount": 4}),
        ))
        assert targets["E1"].fields["damage_taken"] == 0.0

    def test_life_delta_keeps_its_sign(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.LIFE_CHANGE, subjects=("P1",),
                  params={"delta": -3}),
        ))
        assert targets["P1"].fields["life_delta"] == -3.0

    @pytest.mark.parametrize("mode", ["LifeReduced", "PayLife", "LifeLost"])
    def test_a_life_loss_mode_reads_its_amount_as_a_loss(self, mode):
        """A run-parameter event writes the amount and names the direction by
        its mode: a rewritten life loss of 3 is a delta of -3, not +3."""
        record = _record(
            RewritePayload(
                incoming=Event(type=EventType.LIFE_CHANGE, subjects=("P1",),
                               params={"delta": 4, "mode": mode}),
                outgoing=Event(type=EventType.LIFE_CHANGE, subjects=("P1",),
                               params={"delta": 3, "mode": mode}),
            ),
            kind=RecordKind.REWRITE, moment=None,
        )
        assert derive_targets(record)["P1"].fields["life_delta"] == -3.0

    @pytest.mark.parametrize(("params", "delta"), [
        ({"delta": 3, "mode": "GainLife"}, 3.0),
        ({"delta": 3, "mode": "LifeGained"}, 3.0),
        ({"delta": -3, "mode": "LifeLost"}, -3.0),
        ({"delta": -3}, -3.0),
    ])
    def test_a_gain_or_an_already_signed_delta_keeps_its_sign(self, params, delta):
        targets = derive_targets(_resolution(
            Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params=params),
        ))
        assert targets["P1"].fields["life_delta"] == delta

    @pytest.mark.parametrize("mode", ["RemoveCounter", "CounterRemoved"])
    def test_a_counter_removal_mode_reads_its_amount_as_a_removal(self, mode):
        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "P1P1", "delta": 2, "mode": mode}),
        ))
        assert targets["E1"].fields["counters_delta_p1p1"] == -2.0

    def test_gaining_hexproof_from_a_quality_sets_the_hexproof_column(self):
        from effects.domain.output_vocabularies import KEYWORDS_CHANGED

        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["Hexproof:Black"]}),
        ))
        vector = targets["E1"].fields["keywords_gained"]
        assert vector[KEYWORDS_CHANGED.index("hexproof")] == 1.0

    def test_a_tracked_counter_type_gets_its_own_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "P1P1", "delta": 2}),
        ))
        assert targets["E1"].fields["counters_delta_p1p1"] == 2.0

    def test_an_untracked_counter_type_falls_into_the_catch_all(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "FLYING", "delta": 1}),
        ))
        assert targets["E1"].fields["counters_delta_other"] == 1.0

    def test_a_pt_change_splits_into_two_fields(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.PT_CHANGE, subjects=("E1",),
                  params={"power_delta": 3, "toughness_delta": -1}),
        ))
        assert targets["E1"].fields["power_delta"] == 3.0
        assert targets["E1"].fields["toughness_delta"] == -1.0

    def test_mana_lands_in_the_colours_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.MANA_PRODUCED, subjects=("P0",),
                  params={"mana_by_color": {"R": 2, "G": 1}}),
        ))
        assert targets["P0"].fields["mana_delta_r"] == 2.0
        assert targets["P0"].fields["mana_delta_g"] == 1.0

    def test_lost_mana_is_negative(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.MANA_LOST, subjects=("P0",),
                  params={"mana_by_color": {"U": 2}}),
        ))
        assert targets["P0"].fields["mana_delta_u"] == -2.0

    def test_poison_reaches_its_own_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.POISON_CHANGE, subjects=("P1",),
                  params={"delta": 2}),
        ))
        assert targets["P1"].fields["poison_delta"] == 2.0


class TestEverySimpleEventType:
    """Each event type the lookup tables handle, pinned field by field.

    The parameters are the ones the collector writes (fixture and review-pilot
    records): ``damage_healed`` names the healed permanent and the damage the
    clause removed, a ``zone_change`` its lower-case zones and its cause, an
    effect card's move into the command zone no ``from_zone``, and
    ``card_drawn`` always its ``count``. ``control_change`` is hand-written: no
    fixture record holds one.
    """

    @pytest.mark.parametrize(("event_type", "params", "expected"), [
        (EventType.DAMAGE_DEALT,
         {"amount": 3, "combat": False, "damage_type": "normal", "cause": "E126"},
         {"damage_taken": 3.0}),
        (EventType.DAMAGE_HEALED, {"amount": 2, "cause": "E29"}, {"damage_taken": -2.0}),
        (EventType.LIFE_CHANGE, {"delta": -4}, {"life_delta": -4.0}),
        (EventType.POISON_CHANGE, {"delta": 1}, {"poison_delta": 1.0}),
        (EventType.CARD_DRAWN, {"count": 1}, {"cards_drawn": 1.0}),
        (EventType.CARD_DRAWN, {"count": 3}, {"cards_drawn": 3.0}),
        (EventType.CARD_DISCARDED, {"count": 1}, {"cards_discarded": 1.0}),
        (EventType.CARD_MILLED, {"count": 2, "cause": "E75"}, {"cards_milled": 2.0}),
        (EventType.TAPPED, {}, {"tap_state": 1}),
        (EventType.UNTAPPED, {}, {"tap_state": 0}),
        (EventType.CONTROL_CHANGE, {}, {"control_change": 1}),
        (EventType.ATTACHED, {"attached_to": "E22", "cause": "E9"}, {"attached_to_change": 1}),
        (EventType.UNATTACHED, {"detached_from": "E43"}, {"attached_to_change": 1}),
        (EventType.DESTROYED, {}, {"zone_outcome": ZONE_OUTCOMES.index("died")}),
        (EventType.FACE_CHANGE, {}, {"zone_outcome": ZONE_OUTCOMES.index("face_changed")}),
        (EventType.ZONE_CHANGE,
         {"from_zone": "battlefield", "to_zone": "exile", "cause": "E126"},
         {"zone_outcome": ZONE_OUTCOMES.index("exiled")}),
        (EventType.ZONE_CHANGE, {"to_zone": "command", "cause": "E126"},
         {"zone_outcome": ZONE_OUTCOMES.index("to_command")}),
    ])
    def test_its_fields(self, event_type, params, expected):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=("E1",), params=params),
        ))
        assert targets["E1"].fields == expected

    @pytest.mark.parametrize(("event_type", "flag"), [
        (EventType.CARD_LOOKED_AT, "scry"),
        (EventType.LIBRARY_REORDERED, "reorder"),
    ])
    def test_its_library_flag(self, event_type, flag):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=("P0",), params={}),
        ))
        expected = [0.0] * len(LIBRARY_EVENTS)
        expected[LIBRARY_EVENTS.index(flag)] = 1.0
        assert targets["P0"].fields == {"library_events": expected}

    def test_heals_and_hits_on_one_entity_net(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 5}),
            Event(type=EventType.DAMAGE_HEALED, subjects=("E1",),
                  params={"amount": 2, "cause": "E29"}),
        ))
        assert targets["E1"].fields == {"damage_taken": 3.0}
