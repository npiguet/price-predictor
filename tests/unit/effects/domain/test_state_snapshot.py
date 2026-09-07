"""The snapshot's reader rules (T024).

Three of these are the ones a collector or a consumer gets wrong quietly: an
absent tier read as an empty board, the two grant channels merged, and a
perspective baked in at collection time.
"""

from __future__ import annotations

import pytest

from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import (
    COLORS,
    STAGE_ONE_TIERS,
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    PlayerState,
    PowerToughness,
    Refs,
    StackExtras,
    StateSnapshot,
)


@pytest.fixture
def key() -> ProvenanceKey:
    return ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "static", 0)


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
        "entities": (),
    }
    defaults.update(overrides)
    return StateSnapshot(**defaults)


class TestInclusionTiers:
    def test_the_tiers_apply_in_the_documented_order(self):
        assert list(InclusionTier) == [
            InclusionTier.REFERENCED,
            InclusionTier.CORE,
            InclusionTier.UNREFERENCED_STACK,
            InclusionTier.UNREFERENCED_HAND_GRAVEYARD,
        ]

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

    def test_tiers_compare_in_stage_order(self):
        assert InclusionTier.REFERENCED < InclusionTier.CORE
        assert InclusionTier.CORE < InclusionTier.UNREFERENCED_STACK
        assert (
            InclusionTier.UNREFERENCED_STACK
            < InclusionTier.UNREFERENCED_HAND_GRAVEYARD
        )


class TestGrantChannelsStaySeparate:
    def test_an_entity_carries_the_two_channels_in_different_fields(self, key):
        entity = EntityState(
            id="E12", name="Serra Angel", zone="battlefield", controller="P0",
            granted_attached=(key,),
            granted_temporary=GrantedTemporary(keywords=("flying",)),
        )
        assert entity.granted_attached == (key,)
        assert entity.granted_temporary.keywords == ("flying",)

    def test_a_temporary_grant_may_resolve_to_a_line_or_to_a_bare_keyword(
        self, key,
    ):
        """"gains flying until end of turn" names no line; an anthem's does."""
        grant = GrantedTemporary(keywords=("flying",), abilities=(key,))
        assert grant.keywords == ("flying",)
        assert grant.abilities == (key,)

    def test_the_channels_do_not_share_a_default(self):
        first = EntityState(id="E1", name="A", zone="battlefield", controller="P0")
        second = EntityState(id="E2", name="B", zone="battlefield", controller="P0")
        assert first.granted_temporary == second.granted_temporary
        assert first.granted_attached == () == second.granted_attached

    def test_printed_lines_are_a_third_channel_of_their_own(self, key):
        entity = EntityState(
            id="E12", name="Serra Angel", zone="battlefield", controller="P0",
            printed=(key,),
        )
        assert entity.printed == (key,)
        assert entity.granted_attached == ()


class TestPerspectiveIsNotStored:
    def test_controllers_are_absolute_player_ids(self):
        entity = EntityState(
            id="E12", name="Serra Angel", zone="battlefield", controller="P1",
        )
        assert entity.controller == "P1"

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

    def test_the_colour_keys_cover_wubrg_plus_colourless(self):
        assert COLORS == ("W", "U", "B", "R", "G", "C")


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


class TestBlocks:
    def test_the_global_block_carries_the_combat_substep(self):
        assert _snapshot().global_.combat_substep == "first_strike"

    def test_refs_default_to_empty_rather_than_absent(self):
        assert _snapshot().refs == Refs()

    def test_pending_event_is_absent_outside_rewrite_and_trigger_records(self):
        assert _snapshot().pending_event is None

    def test_stack_extras_are_absent_off_the_stack(self):
        entity = EntityState(id="E1", name="A", zone="battlefield", controller="P0")
        assert entity.stack_extras is None

    def test_a_stack_object_carries_its_announced_amounts(self):
        extras = StackExtras(
            targets=("E1", "E2"), per_target_amounts={"E1": 2, "E2": 1},
        )
        entity = EntityState(
            id="S1", name="Fireball", zone="stack", controller="P0",
            stack_extras=extras,
        )
        assert entity.stack_extras.per_target_amounts == {"E1": 2, "E2": 1}

    def test_combat_status_records_becoming_blocked_separately_from_blockers(self):
        """An attacker stays blocked even after every blocker leaves combat."""
        status = CombatStatus(attacking="P1", blocked_by=(), became_blocked=True)
        assert status.blocked_by == ()
        assert status.became_blocked
