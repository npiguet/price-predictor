"""Gate 2's applicability rules, effect by effect (FR-120, FR-122).

Each row's effects say when they are observable, and the row's ``qualifies`` is
derived from them: a combat is an observation of a keyword exactly when one of
its effects applies and has a subject on the board. Testing the two together is
the point — a rule that qualified a combat none of its effects could be read on
would count toward the 200-record threshold and then score as a disagreement
every time, which is a canary that sings whatever the model does.

What each rule encodes is damage actually changing hands. A first-strike hit
that does not kill changes nothing, because the blocker hits back in the regular
step either way; a second hit from a double striker reaches only a creature that
lived through the first; and a creature with no power deals no damage at all.

The boards are deliberately tiny — one attacker, one or two blockers — because
every rule these predicates encode is a rule about two creatures.
"""

from __future__ import annotations

import dataclasses

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    CombatParticipant,
    combat_participant,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
    StateSnapshot,
)

_WHITE_KNIGHT = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "static", 0)


def _creature(
    entity_id: str,
    power: int,
    toughness: int,
    *,
    controller: str = "P0",
    damage: int = 0,
    keywords: tuple[str, ...] = (),
    printed: tuple[ProvenanceKey, ...] = (),
    combat: CombatStatus | None = None,
) -> EntityState:
    return EntityState(
        id=entity_id, name=entity_id, zone="battlefield", controller=controller,
        types=("creature",), pt=PowerToughness(base=(power, toughness)),
        damage=damage, combat=combat, printed=printed,
        granted_temporary=GrantedTemporary(keywords=keywords),
    )


def _board(*entities: EntityState) -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=4, phase="combat_damage", active="P0", priority="P0",
            stack_size=0, combat_substep="regular",
        ),
        players=(
            PlayerState(id="P0", life=20, hand=3, library=30, graveyard=2),
            PlayerState(id="P1", life=17, hand=4, library=28, graveyard=1),
        ),
        entities=entities,
    )


def _attacker_and_blockers(
    *,
    attacker: tuple[int, int] = (2, 2),
    blockers: tuple[tuple[int, int], ...] = ((2, 2),),
    attacker_damage: int = 0,
    attacker_keywords: tuple[str, ...] = (),
    blocker_damage: int = 0,
    blocker_keywords: tuple[str, ...] = (),
) -> tuple[StateSnapshot, EntityState]:
    blocker_ids = tuple(f"B{i}" for i in range(len(blockers)))
    carrier = _creature(
        "A", *attacker, damage=attacker_damage, keywords=attacker_keywords,
        combat=CombatStatus(
            attacking="P1", blocked_by=blocker_ids,
            became_blocked=bool(blocker_ids),
        ),
    )
    blocking = tuple(
        _creature(
            blocker_id, *stats, controller="P1", damage=blocker_damage,
            keywords=blocker_keywords, combat=CombatStatus(blocking=("A",)),
        )
        for blocker_id, stats in zip(blocker_ids, blockers)
    )
    return _board(carrier, *blocking), carrier


def _planeswalker() -> EntityState:
    """A defender that is not a player. Forge writes its name into
    ``CombatStatus.attacking`` exactly as it writes a player's."""
    return EntityState(
        id="E9", name="Chandra", zone="battlefield", controller="P1",
        types=("planeswalker",),
    )


def _unblocked_attacker(power: int = 3, toughness: int = 3):
    carrier = _creature(
        "A", power, toughness, combat=CombatStatus(attacking="P1"),
    )
    return _board(carrier), carrier


def _overlay_keywords(entity) -> set[str]:
    return set(entity.granted_temporary.keywords)


def _printed_first_strike(entity) -> set[str]:
    found = set(entity.granted_temporary.keywords)
    if _WHITE_KNIGHT in entity.printed:
        found.add("first_strike")
    return found


def _qualifies(keyword: str, state, carrier, keywords_of=_overlay_keywords) -> bool:
    participant = combat_participant(state, carrier)
    assert participant is not None
    return KEYWORDS_BY_NAME[keyword].qualifies(participant, keywords_of)


def _observable(
    keyword: str, state, carrier, keywords_of=_overlay_keywords,
) -> list[str]:
    """The labels of the effects this combat can actually show."""
    participant = combat_participant(state, carrier)
    assert participant is not None
    return [
        effect.label for effect in KEYWORDS_BY_NAME[keyword].effects
        if effect.observable(participant, keywords_of)
    ]


class TestCombatParticipant:
    """Who the carrier is fighting, and which player the attack is aimed at."""

    def test_an_attacker_faces_every_creature_blocking_it(self):
        state, carrier = _attacker_and_blockers(blockers=((1, 1), (2, 2)))
        participant = combat_participant(state, carrier)
        assert [o.id for o in participant.opponents] == ["B0", "B1"]

    def test_an_attackers_controller_and_defending_player_are_the_two_sides(self):
        state, carrier = _attacker_and_blockers()
        participant = combat_participant(state, carrier)
        assert participant.controller == "P0"
        assert participant.defending_player == "P1"

    def test_an_unblocked_attacker_has_no_opponent_but_still_has_a_defender(self):
        state, carrier = _unblocked_attacker()
        participant = combat_participant(state, carrier)
        assert participant.opponents == ()
        assert participant.defending_player == "P1"

    def test_a_blocker_faces_the_attacker_it_blocks(self):
        state, _ = _attacker_and_blockers()
        blocker = state.entity("B0")
        participant = combat_participant(state, blocker)
        assert [o.id for o in participant.opponents] == ["A"]
        assert participant.controller == "P1"

    def test_a_blocker_has_no_defending_player(self):
        """Its damage lands on the attacker, never on a player."""
        state, _ = _attacker_and_blockers()
        participant = combat_participant(state, state.entity("B0"))
        assert participant.defending_player is None

    def test_the_defending_player_survives_forges_defender_spelling(self):
        """Forge writes the defender object, not a player id."""
        carrier = _creature(
            "A", 2, 2, combat=CombatStatus(attacking="Ai(1)-Sam"),
        )
        participant = combat_participant(_board(carrier), carrier)
        assert participant.defending_player == "P1"

    def test_a_creature_outside_combat_is_no_participant(self):
        carrier = _creature("A", 2, 2)
        assert combat_participant(_board(carrier), carrier) is None

    def test_a_blocker_whose_attacker_left_the_board_has_no_opponent(self):
        carrier = _creature(
            "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        participant = combat_participant(_board(carrier), carrier)
        assert participant.opponents == ()

    def test_an_attack_on_a_planeswalker_has_no_defending_player(self):
        """Its damage lands on the planeswalker, not on that player's life."""
        carrier = _creature("A", 4, 4, combat=CombatStatus(attacking="Chandra"))
        participant = combat_participant(
            _board(carrier, _planeswalker()), carrier,
        )
        assert participant.defending_player is None

    def test_trample_over_a_planeswalkers_blockers_is_not_an_observation(self):
        carrier = _creature(
            "A", 4, 4,
            combat=CombatStatus(attacking="Chandra", blocked_by=("B0",),
                                became_blocked=True),
        )
        blocker = _creature(
            "B0", 1, 1, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        state = _board(carrier, _planeswalker(), blocker)
        assert not _qualifies("trample", state, carrier)


class TestDealsDamageAtAll:
    """A creature with no power deals none, whatever else is true of it."""

    @pytest.mark.parametrize(
        "keyword",
        ["first_strike", "double_strike", "deathtouch", "lifelink", "wither",
         "infect"],
    )
    def test_a_powerless_carrier_qualifies_for_nothing(self, keyword):
        carrier = _creature(
            "A", 0, 3, keywords=(keyword,),
            combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                                became_blocked=True),
        )
        blocker = _creature(
            "B0", 1, 1, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        assert not _qualifies(keyword, _board(carrier, blocker), carrier)

    def test_a_powerless_carrier_can_still_be_indestructible(self):
        """Indestructible is about damage coming in, not damage going out."""
        state, carrier = _attacker_and_blockers(
            attacker=(0, 2), blockers=((3, 3),),
        )
        assert _qualifies("indestructible", state, carrier)


class TestFirstStrike:
    def test_a_hit_that_kills_a_normal_timing_blocker_qualifies(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 2),),
        )
        assert _qualifies("first_strike", state, carrier)

    def test_a_blocker_the_hit_does_not_kill_does_not_qualify(self):
        """It hits back in the regular step either way, so nothing moves."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 5),),
        )
        assert not _qualifies("first_strike", state, carrier)

    def test_damage_already_on_the_blocker_can_bring_it_into_range(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 4),), blocker_damage=2,
        )
        assert _qualifies("first_strike", state, carrier)

    def test_an_opponent_with_first_strike_disqualifies_it(self):
        """Both hitting first is the same damage step again."""
        state, carrier = _attacker_and_blockers(
            blocker_keywords=("first_strike",),
        )
        assert not _qualifies("first_strike", state, carrier)

    def test_an_opponent_with_double_strike_disqualifies_it(self):
        state, carrier = _attacker_and_blockers(
            blocker_keywords=("double_strike",),
        )
        assert not _qualifies("first_strike", state, carrier)

    def test_several_blockers_make_the_assignment_unpredictable(self):
        """Forge decides how a 3/3 splits three damage over two 2/2s, and the
        table cannot, so neither carrier effect applies."""
        state, carrier = _attacker_and_blockers(
            attacker=(3, 3), blockers=((2, 2), (2, 2)),
        )
        assert _observable("first_strike", state, carrier) == []
        assert not _qualifies("first_strike", state, carrier)

    def test_a_deathtouch_carrier_kills_whatever_it_hits(self):
        """One damage is lethal, so striking first saves even a 1/1."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 1), attacker_keywords=("first_strike", "deathtouch"),
            blockers=((2, 2),),
        )
        assert _qualifies("first_strike", state, carrier)

    def test_a_deathtouch_blocker_is_no_easier_to_kill(self):
        """Deathtouch on the blocker makes the blocker deadlier, not frailer:
        a 1/1 still fails to kill a 2/2 first, so nothing is saved."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 1), blockers=((2, 2),),
            blocker_keywords=("deathtouch",),
        )
        assert not _qualifies("first_strike", state, carrier)

    def test_a_blocker_with_no_power_could_never_have_killed_it(self):
        """It kills the wall in the first step and takes nothing either way."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((0, 2),),
        )
        assert _observable("first_strike", state, carrier) == []

    def test_a_printed_keyword_on_the_opponent_counts_too(self):
        """Printed first strike is most first strike; the overlay sees none."""
        carrier = _creature(
            "A", 2, 2, combat=CombatStatus(attacking="P1", blocked_by=("B0",)),
        )
        blocker = _creature(
            "B0", 2, 2, controller="P1", printed=(_WHITE_KNIGHT,),
            combat=CombatStatus(blocking=("A",)),
        )
        state = _board(carrier, blocker)
        assert not _qualifies(
            "first_strike", state, carrier, keywords_of=_printed_first_strike,
        )
        assert _qualifies("first_strike", state, carrier)

    def test_an_unblocked_attacker_does_not_qualify(self):
        """Nothing strikes back, so striking first changes nothing."""
        state, carrier = _unblocked_attacker()
        assert not _qualifies("first_strike", state, carrier)

    def test_it_moves_the_carriers_damage_and_survival(self):
        state, carrier = _attacker_and_blockers()
        assert _observable("first_strike", state, carrier) == [
            "carrier.damage_taken", "carrier.zone_outcome/died",
        ]


class TestDoubleStrike:
    """Three groups, and which of them a combat can show is the whole rule."""

    def test_a_first_hit_that_kills_shows_the_carrier_effects_only(self):
        """The blocker dies to the first hit, so the second reaches nothing."""
        state, carrier = _attacker_and_blockers(
            attacker=(3, 3), blockers=((2, 2),),
        )
        assert _observable("double_strike", state, carrier) == [
            "carrier.damage_taken", "carrier.zone_outcome/died",
        ]

    def test_a_blocker_that_survives_shows_the_opponent_effects_only(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 5),),
        )
        assert _observable("double_strike", state, carrier) == [
            "opponent.damage_taken", "opponent.zone_outcome/died",
        ]

    def test_an_unblocked_double_striker_qualifies_through_the_player(self):
        state, carrier = _unblocked_attacker()
        assert _observable("double_strike", state, carrier) == [
            "defending_player.life_delta",
        ]

    def test_a_blocker_whose_attacker_is_gone_shows_nothing(self):
        carrier = _creature(
            "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        assert not _qualifies("double_strike", _board(carrier), carrier)

    def test_several_blockers_show_nothing(self):
        state, carrier = _attacker_and_blockers(
            attacker=(3, 3), blockers=((2, 2), (2, 2)),
        )
        assert _observable("double_strike", state, carrier) == []

    def test_a_deathtouch_double_striker_kills_with_its_first_hit(self):
        """So the second hit reaches nothing, whatever the blocker toughness."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 1), attacker_keywords=("double_strike", "deathtouch"),
            blockers=((2, 9),),
        )
        assert _observable("double_strike", state, carrier) == [
            "carrier.damage_taken", "carrier.zone_outcome/died",
        ]


class TestLifelink:
    def test_a_blocked_attacker_deals_damage(self):
        state, carrier = _attacker_and_blockers()
        assert _qualifies("lifelink", state, carrier)

    def test_an_unblocked_attacker_deals_damage(self):
        state, carrier = _unblocked_attacker()
        assert _qualifies("lifelink", state, carrier)

    def test_a_blocker_deals_damage(self):
        state, _ = _attacker_and_blockers()
        assert _qualifies("lifelink", state, state.entity("B0"))

    def test_a_zero_power_lifelinker_gains_nothing(self):
        carrier = _creature(
            "A", 0, 4, keywords=("lifelink",),
            combat=CombatStatus(attacking="P1"),
        )
        assert not _qualifies("lifelink", _board(carrier), carrier)

    def test_a_blocker_whose_attacker_is_gone_deals_nothing(self):
        carrier = _creature(
            "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        assert not _qualifies("lifelink", _board(carrier), carrier)


class TestWither:
    def test_it_needs_a_creature_to_damage(self):
        state, carrier = _attacker_and_blockers()
        assert _qualifies("wither", state, carrier)

    def test_an_unblocked_attacker_does_not_qualify(self):
        """Wither only changes how damage to a creature arrives."""
        state, carrier = _unblocked_attacker()
        assert not _qualifies("wither", state, carrier)


class TestInfect:
    """The creature half and the player half are mutually exclusive."""

    def test_a_blocked_infecter_shows_the_creature_effects_only(self):
        state, carrier = _attacker_and_blockers()
        assert _observable("infect", state, carrier) == [
            "opponent.counters_delta_m1m1", "opponent.damage_taken",
        ]

    def test_an_unblocked_infecter_shows_the_player_effects_only(self):
        state, carrier = _unblocked_attacker()
        assert _observable("infect", state, carrier) == [
            "defending_player.poison_delta", "defending_player.life_delta",
        ]

    def test_a_blocker_shows_the_creature_effects_only(self):
        state, _ = _attacker_and_blockers()
        assert _observable("infect", state, state.entity("B0")) == [
            "opponent.counters_delta_m1m1", "opponent.damage_taken",
        ]

    def test_the_player_loses_less_life_rather_than_more(self):
        """Poison replaces the life loss, so life_delta moves upward."""
        life = next(
            effect for effect in KEYWORDS_BY_NAME["infect"].effects
            if effect.field == "life_delta"
        )
        assert life.direction == 1


class TestDeathtouch:
    def test_an_opponent_that_would_survive_qualifies(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),),
        )
        assert _qualifies("deathtouch", state, carrier)

    def test_an_opponent_that_would_die_anyway_does_not(self):
        state, carrier = _attacker_and_blockers(
            attacker=(3, 3), blockers=((1, 3),),
        )
        assert not _qualifies("deathtouch", state, carrier)

    def test_damage_already_marked_counts_against_the_opponents_toughness(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),), blocker_damage=2,
        )
        assert not _qualifies("deathtouch", state, carrier)

    def test_several_blockers_make_the_assignment_unpredictable(self):
        """Which blocker the two damage goes to is Forge's choice, not a rule."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 1), (1, 5)),
        )
        assert not _qualifies("deathtouch", state, carrier)


class TestTrample:
    def test_a_blocked_attacker_bigger_than_its_blockers_qualifies(self):
        state, carrier = _attacker_and_blockers(
            attacker=(4, 4), blockers=((1, 3),),
        )
        assert _qualifies("trample", state, carrier)

    def test_blockers_that_soak_every_point_do_not_qualify(self):
        state, carrier = _attacker_and_blockers(
            attacker=(4, 4), blockers=((1, 2), (1, 2)),
        )
        assert not _qualifies("trample", state, carrier)

    def test_damage_already_on_a_blocker_leaves_less_to_soak(self):
        state, carrier = _attacker_and_blockers(
            attacker=(4, 4), blockers=((1, 4),), blocker_damage=1,
        )
        assert _qualifies("trample", state, carrier)

    def test_an_unblocked_attacker_does_not_qualify(self):
        """Nothing is in the way, so there is no excess to run over."""
        state, carrier = _unblocked_attacker(power=4)
        assert not _qualifies("trample", state, carrier)

    def test_a_blocker_does_not_qualify(self):
        state, _ = _attacker_and_blockers(attacker=(1, 1), blockers=((4, 4),))
        assert not _qualifies("trample", state, state.entity("B0"))


class TestIndestructible:
    def test_lethal_incoming_damage_qualifies(self):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 3), blockers=((3, 3),),
        )
        assert _qualifies("indestructible", state, carrier)

    def test_survivable_incoming_damage_does_not(self):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 4), blockers=((3, 3),),
        )
        assert not _qualifies("indestructible", state, carrier)

    def test_damage_already_marked_makes_survivable_damage_lethal(self):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 4), blockers=((3, 3),), attacker_damage=1,
        )
        assert _qualifies("indestructible", state, carrier)

    def test_several_blockers_add_up(self):
        """A sum does not depend on how Forge splits the damage, so unlike the
        first-strike rules this one still reads a multi-block."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 4), blockers=((2, 2), (2, 2)),
        )
        assert _qualifies("indestructible", state, carrier)

    def test_a_deathtouch_blocker_makes_any_damage_lethal(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 1),),
            blocker_keywords=("deathtouch",),
        )
        assert _qualifies("indestructible", state, carrier)

    def test_a_powerless_deathtouch_blocker_deals_no_lethal_damage(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((0, 1),),
            blocker_keywords=("deathtouch",),
        )
        assert not _qualifies("indestructible", state, carrier)

    def test_an_unblocked_attacker_takes_nothing(self):
        state, carrier = _unblocked_attacker()
        assert not _qualifies("indestructible", state, carrier)

    def test_an_infect_blocker_kills_through_indestructible(self):
        """Infect and wither damage lands as -1/-1 counters, which shrink the
        carrier to death without destroying it, so the keyword saves nothing."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 2),),
            blocker_keywords=("infect",),
        )
        assert not _qualifies("indestructible", state, carrier)

    def test_a_wither_blocker_kills_through_indestructible(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 2),),
            blocker_keywords=("wither",),
        )
        assert not _qualifies("indestructible", state, carrier)


class TestTheTableIsTheSingleSource:
    def test_every_row_derives_its_predicate_from_its_effects(self):
        for row in KEYWORDS_BY_NAME.values():
            assert callable(row.qualifies), row.keyword
            for effect in row.effects:
                assert callable(effect.applies), f"{row.keyword}/{effect.field}"

    def test_a_row_qualifies_exactly_when_an_effect_is_observable(self):
        state, carrier = _attacker_and_blockers()
        for keyword in KEYWORDS_BY_NAME:
            assert _qualifies(keyword, state, carrier) == bool(
                _observable(keyword, state, carrier)
            ), keyword

    def test_a_row_names_each_head_field_once(self):
        """Double strike reads damage_taken on both sides of the combat."""
        assert KEYWORDS_BY_NAME["double_strike"].fields == (
            "damage_taken", "zone_outcome", "life_delta",
        )

    def test_the_participant_is_frozen(self):
        state, carrier = _attacker_and_blockers()
        participant = combat_participant(state, carrier)
        assert isinstance(participant, CombatParticipant)
        with pytest.raises(dataclasses.FrozenInstanceError):
            participant.controller = "P1"
