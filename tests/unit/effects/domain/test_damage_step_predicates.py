"""Gate 2's applicability rules, effect by effect (FR-120, FR-122).

Each row's effects say when they are observable, and the row's ``qualifies`` is
derived from them: a combat is an observation of a keyword exactly when one of
its effects applies and has a subject on the board. Testing the two together is
the point — a rule that qualified a combat none of its effects could be read on
would count toward the 200-record threshold and then score as a disagreement
every time, which is a canary that sings whatever the model does.

What each rule encodes is damage actually changing hands, inside the one damage
step a combat record covers. First strike and double strike change only the
first-strike step's record — the carrier deals its damage there with the keyword
and none without — so they read what the carrier does to whoever it is fighting
in that step, and never read a regular-step record. A creature with no power
deals no damage at all.

The boards are deliberately tiny — one attacker, one or two blockers — because
every rule these predicates encode is a rule about two creatures.
"""

from __future__ import annotations

import dataclasses

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    CombatParticipant,
    Subject,
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
from tests.unit.effects.builders import fresh_interpreter_env

_WHITE_KNIGHT = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "static", 0)

#: How Forge renders a planeswalker defender into ``CombatStatus.attacking``:
#: its name and its card id, which the snapshot's ``name`` does not carry.
_PLANESWALKER_DEFENDER = "Chandra, Torch of Defiance (9)"

#: How Forge renders a player defender: the player's name, not its ``P`` id.
_PLAYER_DEFENDER = "p2"



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


#: Each damage step's phase, keyed by its substep, as a real record spells them.
_STEPS = {
    "first_strike": "combat_first_strike_damage",
    "regular": "combat_damage",
}


def _board(*entities: EntityState, substep: str = "regular") -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=4, phase=_STEPS[substep], active="P0", priority="P0",
            stack_size=0, combat_substep=substep,
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
    substep: str = "regular",
) -> tuple[StateSnapshot, EntityState]:
    blocker_ids = tuple(f"B{i}" for i in range(len(blockers)))
    carrier = _creature(
        "A", *attacker, damage=attacker_damage, keywords=attacker_keywords,
        combat=CombatStatus(
            attacking=_PLAYER_DEFENDER, blocked_by=blocker_ids,
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
    return _board(carrier, *blocking, substep=substep), carrier


def _planeswalker() -> EntityState:
    """A defender that is not a player."""
    return EntityState(
        id="E9", name="Chandra, Torch of Defiance", zone="battlefield",
        controller="P1", types=("planeswalker",),
    )


def _unblocked_attacker(
    power: int = 3, toughness: int = 3, *, substep: str = "regular",
    keywords: tuple[str, ...] = (),
):
    carrier = _creature(
        "A", power, toughness, keywords=keywords,
        combat=CombatStatus(attacking=_PLAYER_DEFENDER),
    )
    return _board(carrier, substep=substep), carrier


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

    def test_it_carries_the_damage_step_the_record_covers(self):
        state, carrier = _attacker_and_blockers(substep="first_strike")
        assert combat_participant(state, carrier).combat_substep == "first_strike"

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
        carrier = _creature(
            "A", 4, 4, combat=CombatStatus(attacking=_PLANESWALKER_DEFENDER),
        )
        participant = combat_participant(
            _board(carrier, _planeswalker()), carrier,
        )
        assert participant.defending_player is None

    def test_an_attack_on_a_battle_off_the_snapshot_has_no_defending_player(self):
        """The card spelling alone says the defender is not a player."""
        carrier = _creature(
            "A", 4, 4, combat=CombatStatus(attacking="Invasion of Belenon (51)"),
        )
        participant = combat_participant(_board(carrier), carrier)
        assert participant.defending_player is None

    def test_an_attack_on_nothing_has_no_defending_player(self):
        carrier = _creature(
            "A", 4, 4, combat=CombatStatus(attacking="<Nothing>"),
        )
        participant = combat_participant(_board(carrier), carrier)
        assert participant.defending_player is None

    def test_trample_over_a_planeswalkers_blockers_is_not_an_observation(self):
        carrier = _creature(
            "A", 4, 4,
            combat=CombatStatus(
                attacking=_PLANESWALKER_DEFENDER, blocked_by=("B0",),
                became_blocked=True,
            ),
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
    @pytest.mark.parametrize("substep", ["first_strike", "regular"])
    def test_a_powerless_carrier_qualifies_for_nothing(self, keyword, substep):
        carrier = _creature(
            "A", 0, 3, keywords=(keyword,),
            combat=CombatStatus(attacking=_PLAYER_DEFENDER, blocked_by=("B0",),
                                became_blocked=True),
        )
        blocker = _creature(
            "B0", 1, 1, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        state = _board(carrier, blocker, substep=substep)
        assert not _qualifies(keyword, state, carrier)

    def test_a_powerless_carrier_can_still_be_indestructible(self):
        """Indestructible is about damage coming in, not damage going out."""
        state, carrier = _attacker_and_blockers(
            attacker=(0, 2), blockers=((3, 3),),
        )
        assert _qualifies("indestructible", state, carrier)


_STRIKE_FIRST_ROWS = ["first_strike", "double_strike"]


@pytest.mark.parametrize("keyword", _STRIKE_FIRST_ROWS)
class TestStrikingFirst:
    """First strike and double strike, read inside the first-strike step.

    The two rows share every rule: within that step's record the carrier deals
    its damage with the keyword and none without it, so what moves is the
    creature or player on the receiving end of its hit.
    """

    def test_a_hit_on_a_lone_blocker_moves_its_damage(self, keyword):
        """A 2/2 does not kill a 3/5, but the blocker still takes the 2."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 5),), attacker_keywords=(keyword,),
            substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == ["opponent.damage_taken"]

    def test_a_lethal_hit_moves_its_death_too(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 2),), attacker_keywords=(keyword,),
            substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == [
            "opponent.damage_taken", "opponent.zone_outcome/died",
        ]

    def test_damage_already_on_the_blocker_can_bring_it_into_range(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 4),), blocker_damage=2,
            attacker_keywords=(keyword,), substep="first_strike",
        )
        assert "opponent.zone_outcome/died" in _observable(keyword, state, carrier)

    def test_a_deathtouch_carrier_kills_whatever_it_hits(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 1), blockers=((2, 9),),
            attacker_keywords=(keyword, "deathtouch"), substep="first_strike",
        )
        assert "opponent.zone_outcome/died" in _observable(keyword, state, carrier)

    def test_a_deathtouch_blocker_is_no_easier_to_kill(self, keyword):
        """Deathtouch on the blocker makes it deadlier, not frailer."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 1), blockers=((2, 2),), attacker_keywords=(keyword,),
            blocker_keywords=("deathtouch",), substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == ["opponent.damage_taken"]

    @pytest.mark.parametrize("opponent_keyword", _STRIKE_FIRST_ROWS)
    def test_an_opponent_striking_first_too_changes_nothing(
        self, keyword, opponent_keyword,
    ):
        """It deals its own damage in this step either way; the carrier's hit
        lands here with the keyword and not without it all the same."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((2, 2),), attacker_keywords=(keyword,),
            blocker_keywords=(opponent_keyword,), substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == [
            "opponent.damage_taken", "opponent.zone_outcome/died",
        ]

    def test_an_opponent_with_no_power_changes_nothing(self, keyword):
        """Nothing but a first striker deals damage in this step, so whether
        the wall could have hit back is not this record's question."""
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((0, 2),), attacker_keywords=(keyword,),
            substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == [
            "opponent.damage_taken", "opponent.zone_outcome/died",
        ]

    def test_the_carriers_own_damage_is_never_read(self, keyword):
        """In this step the opponent deals the same damage — none, or its own
        first-strike hit — whatever the carrier keeps."""
        for effect in KEYWORDS_BY_NAME[keyword].effects:
            assert effect.subject is not Subject.CARRIER, effect.label

    def test_a_blocker_hits_the_attacker_it_blocks(self, keyword):
        state, _ = _attacker_and_blockers(
            attacker=(3, 3), blockers=((2, 2),), blocker_keywords=(keyword,),
            substep="first_strike",
        )
        blocker = state.entity("B0")
        assert _observable(keyword, state, blocker) == ["opponent.damage_taken"]

    def test_an_unblocked_attacker_moves_the_defending_players_life(self, keyword):
        state, carrier = _unblocked_attacker(
            keywords=(keyword,), substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == [
            "defending_player.life_delta",
        ]

    def test_an_unblocked_attack_on_a_planeswalker_shows_nothing(self, keyword):
        carrier = _creature(
            "A", 4, 4, keywords=(keyword,),
            combat=CombatStatus(attacking=_PLANESWALKER_DEFENDER),
        )
        state = _board(carrier, _planeswalker(), substep="first_strike")
        assert not _qualifies(keyword, state, carrier)

    def test_several_blockers_make_the_assignment_unpredictable(self, keyword):
        """Forge decides how a 3/3 splits three damage over two 2/2s."""
        state, carrier = _attacker_and_blockers(
            attacker=(3, 3), blockers=((2, 2), (2, 2)),
            attacker_keywords=(keyword,), substep="first_strike",
        )
        assert _observable(keyword, state, carrier) == []

    def test_a_blocker_whose_attacker_is_gone_shows_nothing(self, keyword):
        carrier = _creature(
            "B0", 2, 2, controller="P1", keywords=(keyword,),
            combat=CombatStatus(blocking=("A",)),
        )
        state = _board(carrier, substep="first_strike")
        assert not _qualifies(keyword, state, carrier)

    @pytest.mark.parametrize("shape", ["blocked", "unblocked", "blocker"])
    def test_a_regular_step_record_never_qualifies(self, keyword, shape):
        """The first strike already landed; the regular step's board carries
        it, so nothing in that record turns on the keyword."""
        if shape == "unblocked":
            state, carrier = _unblocked_attacker(keywords=(keyword,))
        elif shape == "blocked":
            state, carrier = _attacker_and_blockers(
                attacker=(2, 2), blockers=((2, 2),),
                attacker_keywords=(keyword,),
            )
        else:
            state, _ = _attacker_and_blockers(
                attacker=(2, 2), blockers=((2, 2),),
                blocker_keywords=(keyword,),
            )
            carrier = state.entity("B0")
        assert state.global_.combat_substep == "regular"
        assert _observable(keyword, state, carrier) == []

    def test_a_record_with_no_substep_never_qualifies(self, keyword):
        state, carrier = _unblocked_attacker(
            keywords=(keyword,), substep="first_strike",
        )
        state = dataclasses.replace(
            state,
            global_=dataclasses.replace(state.global_, combat_substep=None),
        )
        assert not _qualifies(keyword, state, carrier)


class TestTheOtherStrikeFirstKeyword:
    """Stripping one keyword from a creature the other still puts in the
    first-strike step moves nothing, so the row does not read it."""

    @pytest.mark.parametrize(
        "stripped, kept",
        [("first_strike", "double_strike"), ("double_strike", "first_strike")],
    )
    def test_the_other_keyword_keeps_it_striking_first(self, stripped, kept):
        state, carrier = _unblocked_attacker(
            keywords=(stripped, kept), substep="first_strike",
        )
        assert not _qualifies(stripped, state, carrier)

    def test_a_printed_first_strike_is_found_through_the_resolver(self):
        """Printed first strike is most first strike; the overlay sees none."""
        carrier = _creature(
            "A", 2, 2, keywords=("double_strike",), printed=(_WHITE_KNIGHT,),
            combat=CombatStatus(attacking=_PLAYER_DEFENDER),
        )
        state = _board(carrier, substep="first_strike")
        assert _qualifies("double_strike", state, carrier)
        assert not _qualifies(
            "double_strike", state, carrier, keywords_of=_printed_first_strike,
        )


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
            combat=CombatStatus(attacking=_PLAYER_DEFENDER),
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

    @pytest.mark.parametrize("substep", ["first_strike", "regular"])
    def test_a_row_qualifies_exactly_when_an_effect_is_observable(self, substep):
        state, carrier = _attacker_and_blockers(substep=substep)
        for keyword in KEYWORDS_BY_NAME:
            assert _qualifies(keyword, state, carrier) == bool(
                _observable(keyword, state, carrier)
            ), keyword

    @pytest.mark.parametrize(
        "keyword",
        ["deathtouch", "lifelink", "trample", "indestructible", "wither",
         "infect"],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            {"attacker": (4, 2), "blockers": ((1, 3),)},
            {"attacker": (1, 2), "blockers": ((3, 3),)},
        ],
    )
    def test_the_other_rows_read_either_damage_step_alike(self, keyword, shape):
        """Only the two strike-first rows turn on which step a record covers."""
        observed = []
        for substep in ("first_strike", "regular"):
            state, carrier = _attacker_and_blockers(**shape, substep=substep)
            observed.append(_observable(keyword, state, carrier))
        unblocked = [
            _observable(keyword, *_unblocked_attacker(substep=substep))
            for substep in ("first_strike", "regular")
        ]
        assert observed[0] == observed[1]
        assert unblocked[0] == unblocked[1]

    def test_a_row_names_each_head_field_once(self):
        """Infect reads two fields on the creature and two on the player."""
        assert KEYWORDS_BY_NAME["infect"].fields == (
            "counters_delta_m1m1", "damage_taken", "poison_delta", "life_delta",
        )

    def test_the_participant_is_frozen(self):
        state, carrier = _attacker_and_blockers()
        participant = combat_participant(state, carrier)
        assert isinstance(participant, CombatParticipant)
        with pytest.raises(dataclasses.FrozenInstanceError):
            participant.controller = "P1"


def test_the_keyword_predicate_module_does_not_import_torch():
    """build-corpus survey workers use it (FR-045) and must not load torch."""
    import subprocess
    import sys

    code = (
        "import sys; import effects.domain.damage_step_keywords as m; "
        "assert m.KeywordResolver and m.qualifying_observations; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    child = subprocess.run([sys.executable, "-c", code], env=fresh_interpreter_env())
    assert child.returncode == 0
