"""The ``[GLOBAL]``, ``[ACT]`` and ``[PLAYER]`` slots: controller tags derive
from the record's actor rather than being stored, scalars enter unbinned as
raw plus a log1p copy, the player, global and act inputs, and the targets
read as cast."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from effects.domain.card_features import card_features
from effects.domain.event_schema import Event, EventType
from effects.domain.input_layout import (
    ACT_SLOT,
    THERMOMETER_STEPS,
    act_features,
    global_features,
    player_features,
    push_scalar,
)
from effects.domain.input_vocabularies import PHASES
from effects.domain.records import (
    RecordKind,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    GlobalState,
    PlayerState,
    Refs,
)
from tests.unit.effects.domain.effect_head_input_support import (
    SlotLink,
    _build,
    _gen3_entity,
    _gen3_record,
    _linked,
    _record,
    _snapshot,
    _subject_slot,
)


class TestControllerTags:
    def test_mine_and_opponent_derive_from_the_records_actor(self):
        state = _snapshot()
        from_p0 = card_features(state.entities[0], _record(state=state))
        from_p1 = card_features(
            state.entities[0], _record(state=state, actor_player="P1"),
        )
        assert from_p0 != from_p1

    def test_the_player_slot_flags_the_actor(self):
        record = _record()
        mine = player_features(record.state.players[0], record)
        theirs = player_features(record.state.players[1], record)
        # Acting, not acting, active, holding priority and targeted sit just
        # before the life thermometer; P0 is active and holds priority.
        flags = slice(-5 - THERMOMETER_STEPS, -THERMOMETER_STEPS)
        assert mine[flags] == (1.0, 0.0, 1.0, 1.0, 0.0)
        assert theirs[flags] == (0.0, 1.0, 0.0, 0.0, 0.0)

    def test_a_targeted_player_is_flagged(self):
        state = _snapshot(refs=Refs(targets=("P1",)))
        record = _record(state=state)
        assert player_features(state.players[1], record)[-1 - THERMOMETER_STEPS] == 1.0

    def test_the_global_slot_flags_whose_turn_it_is(self):
        mine = global_features(_record())
        theirs = global_features(_record(actor_player="P1"))
        assert mine != theirs


class TestNumericEncoding:
    @staticmethod
    def _scalar(value: float) -> list[float]:
        out: list[float] = []
        push_scalar(out, value)
        return out

    def test_a_scalar_enters_raw_plus_log1p(self):
        assert self._scalar(3) == pytest.approx([3.0, math.log1p(3)])

    def test_the_log_copy_is_signed_so_it_stays_monotone_across_zero(self):
        assert self._scalar(-3)[1] == pytest.approx(-math.log1p(3))
        assert self._scalar(-3)[1] < self._scalar(0)[1] < self._scalar(3)[1]

    def test_nothing_is_binned(self):
        """Two nearby values give two different feature vectors."""
        assert self._scalar(3) != self._scalar(4)

    def test_a_players_life_reaches_the_features_unbinned(self):
        record = _record()
        low = player_features(
            PlayerState(id="P0", life=1, hand=0, library=0, graveyard=0), record,
        )
        high = player_features(
            PlayerState(id="P0", life=2, hand=0, library=0, graveyard=0), record,
        )
        assert low != high


class TestPlayerGlobalAndActInputs:
    """Root spec § 9.4-§ 9.5 and the land-play limit of FR-044."""

    def test_max_land_plays_beside_lands_played_and_unlimited(self):
        from effects.domain.input_vocabularies import THIS_TURN_COUNTERS

        record = _record()
        at = 6 * 2 + len(THIS_TURN_COUNTERS) * 2

        def limit(player):
            return player_features(player, record)[at:at + 3]

        two = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0, max_land_plays=2)
        free = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0,
                           land_plays_unlimited=True)
        assert limit(two) == pytest.approx((2.0, math.log1p(2), 0.0))
        assert limit(free) == (0.0, 0.0, 1.0)

    def test_player_designations_reach_the_slot(self):
        record = _record()
        plain = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0)
        for change in ({"speed": 2}, {"rad": 3}, {"monarch": True},
                       {"initiative": True}, {"city_blessing": True}):
            assert player_features(replace(plain, **change), record) != player_features(
                plain, record), change

    def test_day_night_and_the_pending_event_reach_the_global_slot(self):
        from effects.domain.input_vocabularies import DAY_NIGHT, EVENT_TYPES

        night = _record(state=_snapshot(global_=GlobalState(
            turn=2, phase="main1", active="P0", priority="P0", stack_size=0,
            day_night="night")))
        # turn, stack size, the phase and substep one-hots, the two actor
        # flags and the emblem count come first.
        from effects.domain.input_vocabularies import COMBAT_SUBSTEPS

        start = 4 + len(PHASES) + 1 + len(COMBAT_SUBSTEPS) + 1 + 2 + 2
        assert global_features(night)[start + DAY_NIGHT.index("night")] == 1.0
        pending = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 3})
        record = _record(kind=RecordKind.TRIGGER, moment=None,
                         state=_snapshot(pending_event=pending),
                         payload=TriggerPayload(event=pending, fired=True))
        at = start + len(DAY_NIGHT) + 1
        event = global_features(record)[at:at + len(EVENT_TYPES) + 3]
        assert event[EVENT_TYPES.index("damage_dealt")] == 1.0
        assert event[-2] == 3.0

    def test_dice_and_coins_reach_the_act_slot(self):
        from effects.domain.state_snapshot import RandomResults

        rolled = _record(state=_snapshot(refs=Refs(
            random=RandomResults(dice=(4, 6), coin_flips_won=1))))
        assert act_features(rolled) != act_features(_record())
        assert len(act_features(rolled)) == len(act_features(_record()))


class TestTheTargetsAsCast:
    """An effect half's ``targets`` leave out what became illegal before it
    resolved; the input reads ``targets_at_cast`` so that a fizzle is not
    handed to the model it is asked to predict."""

    @staticmethod
    def _fizzled(**refs):
        return _record(state=_snapshot(refs=Refs(**refs)))

    @staticmethod
    def _relation_to_act(entity, record):
        from effects.domain.card_features import card_features, card_segments

        return card_features(entity, record)[card_segments(entity, record)["relation_to_act"]]

    def test_a_fizzled_half_reads_its_targets_as_cast(self):
        record = self._fizzled(targets=(), targets_at_cast=("E2", "P1"), targets_on_stack=())
        out = self._relation_to_act(record.state.entities[1], record)
        assert out[1] == 1.0
        assert player_features(record.state.players[1], record)[-1 - THERMOMETER_STEPS] == 1.0
        unknown = self._fizzled(targets=("P1",), targets_at_cast=None)
        assert unknown.state.refs.as_cast == ("P1",)

    def test_the_count_of_targets_is_the_count_as_cast(self):
        from effects.domain.input_layout import act_features

        fizzled = act_features(self._fizzled(targets=(), targets_at_cast=("E2",)))
        resolved = act_features(self._fizzled(targets=("E2",), targets_at_cast=("E2",)))
        assert fizzled == resolved

    def test_a_permanent_whose_ability_is_targeted_is_not_itself_targeted(self):
        record = self._fizzled(targets=("E2",), targets_at_cast=("E2",),
                               targets_on_stack=("E2",))
        out = self._relation_to_act(record.state.entities[1], record)
        assert out[1] == 0.0

    def test_the_act_slot_targets_what_it_was_cast_at(self):
        record = _gen3_record(
            (_gen3_entity("E1"), _gen3_entity("E2")),
            refs=Refs(targets=(), targets_at_cast=("E2",), source="E1"),
        )
        surface = _build(record)
        assert _linked(surface, SlotLink(ACT_SLOT, _subject_slot(surface, "E2"), "targets", 0))
