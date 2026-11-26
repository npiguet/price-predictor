"""The input layout reads the collector's own spellings (FR-063c)."""

from __future__ import annotations

from effects.domain.card_features import card_features, card_segments, counter_rows
from effects.domain.event_schema import Event, EventType
from effects.domain.input_layout import (
    THERMOMETER_STEPS,
    global_features,
    player_features,
)
from effects.domain.input_vocabularies import PHASES
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.state_snapshot import (
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
)
from tests.unit.effects.domain.effect_head_input_support import (
    _entity,
    _record,
    _snapshot,
)


class TestCollectorSpellings:
    """The input layout reads the collector's names and compares amounts (FR-063c).

    The spellings are the collector's own, as a real shard writes them:
    Forge's ``combat_*`` phase names, ``+1/+1`` counters, and keywords already
    normalized at parse time (``protection_from_red``, ``landwalk``).
    """

    @staticmethod
    def _phase_slot(record):
        features = global_features(record)
        # turn and stack size are two scalars each; the phase one-hot follows.
        return features[4:4 + len(PHASES) + 1]

    def test_a_forge_combat_phase_reaches_its_own_slot(self):
        state = _snapshot(global_=GlobalState(
            turn=3, phase="combat_declare_attackers", active="P0",
            priority="P0", stack_size=0,
        ))
        record = _record(state=state)
        assert self._phase_slot(record)[PHASES.index("declare_attackers")] == 1.0

    def test_a_pending_life_loss_reads_as_a_loss(self):
        """A trigger's map writes the amount and names the direction by its
        mode: ``LifeLost 3`` is -3 in the pending event's amount, ``LifeGained 3`` +3."""
        def amounts(mode: str) -> tuple[float, ...]:
            pending = Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3, "mode": mode})
            return global_features(_record(state=_snapshot(pending_event=pending)))

        lost, gained = amounts("LifeLost"), amounts("LifeGained")
        differ = [i for i, (a, b) in enumerate(zip(lost, gained, strict=True)) if a != b]
        assert [lost[i] for i in differ][:1] == [-3.0]
        assert [gained[i] for i in differ][:1] == [3.0]

    def test_every_forge_combat_phase_name_is_mapped(self):
        for forge_name in (
            "combat_begin", "combat_declare_attackers", "combat_declare_blockers",
            "combat_first_strike_damage", "combat_end",
        ):
            state = _snapshot(global_=GlobalState(
                turn=3, phase=forge_name, active="P0", priority="P0",
                stack_size=0,
            ))
            assert self._phase_slot(_record(state=state))[-1] == 0.0

    def test_plus_one_counters_read_as_p1p1(self):
        from effects.infrastructure.record_codec import _counters_from_json

        decoded = _counters_from_json({"+1/+1": 2})
        assert counter_rows(_entity("E1", counters=decoded)) == counter_rows(
            _entity("E1", counters={"P1P1": 2}))

    def test_protection_from_a_colour_and_landwalk_set_their_flags(self):
        entity = _entity("E1", granted_temporary=GrantedTemporary(
            keywords=("protection_from_red", "landwalk"),
        ))
        record = _record(state=_snapshot(entities=(entity,)))
        block = card_features(entity, record)[card_segments(entity, record)["keyword_flags"]]
        assert block[KEPT_KEYWORD_FLAGS.index("protection")] == 1.0
        assert block[KEPT_KEYWORD_FLAGS.index("landwalk")] == 1.0
        assert sum(block) == 2

    def test_the_card_ends_with_power_toughness_and_remaining_toughness(self):
        entity = _entity(
            "E1", pt=PowerToughness(base=(2, 3), boosts=(1, 1), counters=(1, 1)),
            damage=2,
        )
        record = _record(state=_snapshot(entities=(entity,)))
        features = card_features(entity, record)
        steps = THERMOMETER_STEPS
        power, toughness, remaining = (
            features[-3 * steps:-2 * steps],
            features[-2 * steps:-steps],
            features[-steps:],
        )
        assert sum(power) == 4 and power[3] == 1.0 and power[4] == 0.0
        assert sum(toughness) == 5
        assert sum(remaining) == 3

    def test_an_entity_without_power_and_toughness_carries_zero_codes(self):
        entity = _entity("E1", types=("Enchantment",), pt=None)
        record = _record(state=_snapshot(entities=(entity,)))
        features = card_features(entity, record)
        assert features[-3 * THERMOMETER_STEPS:] == (0.0,) * (3 * THERMOMETER_STEPS)

    def test_the_player_ends_with_a_life_thermometer(self):
        record = _record()
        features = player_features(record.state.players[1], record)
        assert sum(features[-THERMOMETER_STEPS:]) == 8  # P1 is at 8 life

    def test_life_above_the_starting_twenty_reads_in_full(self):
        players = (
            PlayerState(id="P0", life=33, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        )
        record = _record(state=_snapshot(players=players))
        features = player_features(players[0], record)
        assert sum(features[-THERMOMETER_STEPS:]) == 33
