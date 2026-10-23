"""The snapshot's reader rules (T024).

Two of these are the ones a collector or a consumer gets wrong quietly: an
absent tier read as an empty board, and a perspective baked in at collection
time.
"""

from __future__ import annotations

from effects.domain.state_snapshot import (
    STAGE_ONE_TIERS,
    EntityState,
    GlobalState,
    InclusionTier,
    PlayerState,
    PowerToughness,
    StateSnapshot,
)
from tests.unit.effects.builders import make_snapshot


def _snapshot(**overrides) -> StateSnapshot:
    defaults = {
        "global_": GlobalState(
            turn=7, phase="combat_damage", active="P0", priority="P1",
            stack_size=1, combat_substep="first_strike",
        ),
        "players": (
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


class TestInclusionTiers:
    def test_a_snapshot_defaults_to_the_stage_one_tiers(self):
        assert _snapshot().tiers == STAGE_ONE_TIERS

    def test_an_absent_tier_means_uncollected_not_empty(self):
        """A stage-one snapshot says nothing about the stack, empty or not."""
        snapshot = _snapshot()
        assert snapshot.global_.stack_size == 1
        assert snapshot.entities == ()
        assert not snapshot.collected(InclusionTier.UNREFERENCED_STACK)

    def test_a_collected_tier_reports_as_collected(self):
        snapshot = _snapshot(
            tiers=STAGE_ONE_TIERS | {InclusionTier.UNREFERENCED_STACK},
        )
        assert snapshot.collected(InclusionTier.UNREFERENCED_STACK)
        assert not snapshot.collected(InclusionTier.UNREFERENCED_HAND_GRAVEYARD)

class TestPerspectiveIsNotStored:
    def test_mine_and_opponent_derive_from_the_records_actor(self):
        snapshot = _snapshot()
        assert snapshot.controller_tag("P0", actor_player="P0") == "mine"
        assert snapshot.controller_tag("P0", actor_player="P1") == "opponent"

    def test_one_snapshot_serves_records_with_different_actors(self):
        """The reason perspective is derived: the same board, two actors."""
        snapshot = _snapshot()
        assert snapshot.controller_tag("P1", "P0") != snapshot.controller_tag(
            "P1", "P1"
        )

    def test_no_snapshot_field_names_a_perspective(self):
        from dataclasses import fields

        names = {f.name for f in fields(StateSnapshot)}
        assert not names & {"perspective", "viewer", "mine", "opponent"}


class TestCharacteristics:
    def test_power_and_toughness_stay_decomposed(self):
        pt = PowerToughness(base=(2, 2), boosts=(1, 1), counters=(0, 1))
        assert pt.base == (2, 2)
        assert pt.total == (3, 4)

    def test_two_routes_to_the_same_total_stay_distinguishable(self):
        by_boost = PowerToughness(base=(2, 2), boosts=(1, 1))
        by_counter = PowerToughness(base=(2, 2), counters=(1, 1))
        assert by_boost.total == by_counter.total
        assert by_boost != by_counter

class TestLookups:
    def test_an_entity_resolves_by_id(self):
        entity = EntityState(id="E12", name="A", zone="battlefield", controller="P0")
        assert _snapshot(entities=(entity,)).entity("E12") is entity

    def test_a_missing_entity_returns_none(self):
        assert _snapshot().entity("E99") is None

    def test_a_player_resolves_by_id(self):
        assert _snapshot().player("P1").life == 8

    def test_a_missing_player_returns_none(self):
        assert _snapshot().player("P9") is None
