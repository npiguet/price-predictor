"""First strike and double strike, read inside the first-strike step (FR-120, FR-122).

Within that step's record the carrier deals its damage with the keyword and
none without it, so the rows read what the carrier does to whoever it is
fighting there, and never read a regular-step record.
"""

from __future__ import annotations

import dataclasses

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    Subject,
)
from effects.domain.state_snapshot import (
    CombatStatus,
)
from tests.unit.effects.domain.test_damage_step_predicates import (
    _PLANESWALKER_DEFENDER,
    _PLAYER_DEFENDER,
    _WHITE_KNIGHT,
    _attacker_and_blockers,
    _board,
    _creature,
    _observable,
    _planeswalker,
    _qualifies,
    _unblocked_attacker,
)


def _printed_first_strike(entity) -> set[str]:
    found = set(entity.granted_temporary.keywords)
    if _WHITE_KNIGHT in entity.printed:
        found.add("first_strike")
    return found


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
