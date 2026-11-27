"""Where and when the carrier's damage lands, for the rows besides striking first.

No damage changes hands in a first-strike step nobody strikes in; a hit
reaches what the carrier fights only where the rules let it (blockers gone,
protection, phasing); and each row reads the damage of the record's own step
(FR-120, FR-122).
"""

from __future__ import annotations

import dataclasses

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    KeywordResolver,
    qualifying_observations,
)
from effects.domain.state_snapshot import (
    CombatStatus,
)
from tests.unit.effects.domain.test_damage_step_predicates import (
    _PLAYER_DEFENDER,
    _attacker_and_blockers,
    _board,
    _combat_record,
    _creature,
    _observable,
    _participant_of,
    _qualifies,
    _unblocked_attacker,
)


class TestTheTableIsTheSingleSource:
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
    def test_no_damage_changes_hands_in_a_step_nobody_strikes_in(self, keyword, shape):
        """Rule 510.4: creatures without first or double strike deal no damage
        in the first-strike step, so no other row can be read there; in the
        regular step they strike as before."""
        state, carrier = _attacker_and_blockers(**shape, substep="first_strike")
        assert _observable(keyword, state, carrier) == []
        assert _observable(keyword, *_unblocked_attacker(substep="first_strike")) == []


class TestWhatTheHitReaches:
    """Damage the carrier deals reaches what it is fighting only where the
    rules let it: a blocked attacker whose blockers all left deals none
    without trample (509.1h, 510.1c), and protection prevents damage from a
    source with its quality (702.16e)."""

    def _blockerless(self, *keywords: str):
        carrier = _creature("A", 3, 3, keywords=keywords, combat=CombatStatus(
            attacking=_PLAYER_DEFENDER, blocked_by=(), became_blocked=True))
        return _board(carrier), carrier

    @pytest.mark.parametrize("keyword", ["lifelink", "infect", "wither"])
    def test_a_blocked_attacker_whose_blockers_left_deals_no_damage(self, keyword):
        assert not _qualifies(keyword, *self._blockerless(keyword))

    def test_a_blockerless_trampler_assigns_its_damage_to_the_player(self):
        """702.19e."""
        assert _qualifies("trample", *self._blockerless("trample"))

    @pytest.mark.parametrize("keyword", ["lifelink", "deathtouch", "wither", "infect"])
    def test_a_blocker_protected_from_the_carrier_takes_no_damage(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 2), blockers=((1, 3),), attacker_keywords=(keyword,),
            blocker_keywords=("protection_from_red",))
        carrier = dataclasses.replace(carrier, colors=("R",))
        state = dataclasses.replace(state, entities=(carrier, *state.entities[1:]))
        assert not _qualifies(keyword, state, carrier)
        unprotected = dataclasses.replace(carrier, colors=("G",))
        state = dataclasses.replace(state, entities=(unprotected, *state.entities[1:]))
        assert _qualifies(keyword, state, unprotected)

    def test_an_indestructible_carrier_is_not_threatened_by_damage_it_is_protected_from(self):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 3),),
            attacker_keywords=("indestructible", "protection_from_green"))
        blocker = dataclasses.replace(state.entities[1], colors=("G",))
        state = dataclasses.replace(state, entities=(carrier, blocker))
        assert not _qualifies("indestructible", state, carrier)

    def test_a_phased_out_blocker_is_no_opponent_and_a_last_known_copy_no_carrier(self):
        """702.26b: a phased-out permanent is treated as though it does not
        exist; a last-known copy (no zone) is no creature in the combat."""
        state, carrier = _attacker_and_blockers(attacker=(2, 2), blockers=((1, 3),),
                                                attacker_keywords=("lifelink",))
        gone = dataclasses.replace(state.entities[1], designations=("phased_out",))
        state = dataclasses.replace(state, entities=(carrier, gone))
        assert _participant_of(state, carrier).opponents == ()
        ghost = dataclasses.replace(carrier, id="L9", zone=None)
        record = _combat_record(dataclasses.replace(state, entities=(ghost,)))
        assert qualifying_observations(record, (KEYWORDS_BY_NAME["lifelink"],),
                                       KeywordResolver()) == []


class TestTheCarriersOwnStep:
    """The rows besides the strike-first ones read the damage the carrier deals,
    or takes, in the record's step (rules 510.4, 702.7b, 702.4b)."""

    @pytest.mark.parametrize("keyword", ["lifelink", "wither", "infect"])
    def test_a_first_striker_deals_its_damage_in_the_first_strike_step(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),), attacker_keywords=("first_strike", keyword),
            substep="first_strike")
        assert _qualifies(keyword, state, carrier)

    @pytest.mark.parametrize("keyword", ["lifelink", "wither", "infect", "deathtouch"])
    def test_a_first_striker_deals_none_in_the_regular_step(self, keyword):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),), attacker_keywords=("first_strike", keyword),
            substep="regular")
        assert not _qualifies(keyword, state, carrier)

    @pytest.mark.parametrize("substep", ["first_strike", "regular"])
    def test_a_double_striker_deals_damage_in_both_steps(self, substep):
        state, carrier = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),),
            attacker_keywords=("double_strike", "lifelink"), substep=substep)
        assert _qualifies("lifelink", state, carrier)

    def test_a_first_striking_deathtouch_carrier_qualifies_in_its_own_step(self):
        state, carrier = _attacker_and_blockers(
            attacker=(1, 2), blockers=((1, 3),),
            attacker_keywords=("first_strike", "deathtouch"), substep="first_strike")
        assert _qualifies("deathtouch", state, carrier)

    def test_a_trampler_runs_over_only_in_its_own_step(self):
        for substep, strikes in (("first_strike", False), ("regular", True)):
            state, carrier = _attacker_and_blockers(
                attacker=(5, 5), blockers=((1, 3),), attacker_keywords=("trample",),
                substep=substep)
            assert _qualifies("trample", state, carrier) is strikes, substep

    def test_a_deathtouch_trampler_needs_assign_only_one_damage_a_blocker(self):
        """702.2c with 702.19b: with deathtouch one damage is lethal, so a 2-power trampler
        runs over a 1/3 blocker."""
        plain = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),), attacker_keywords=("trample",))
        assert not _qualifies("trample", *plain)
        deadly = _attacker_and_blockers(
            attacker=(2, 2), blockers=((1, 3),), attacker_keywords=("trample", "deathtouch"))
        assert _qualifies("trample", *deadly)

    def test_indestructible_reads_only_the_damage_coming_in_this_step(self):
        """A lethal blocker without first strike deals nothing in the first-strike
        step; a first-striking one does."""
        plain = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 3),), attacker_keywords=("indestructible",),
            substep="first_strike")
        assert not _qualifies("indestructible", *plain)
        first = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 3),), attacker_keywords=("indestructible",),
            blocker_keywords=("first_strike",), substep="first_strike")
        assert _qualifies("indestructible", *first)
        later = _attacker_and_blockers(
            attacker=(2, 2), blockers=((3, 3),), attacker_keywords=("indestructible",),
            blocker_keywords=("first_strike",), substep="regular")
        assert not _qualifies("indestructible", *later)

    def test_an_indestructible_opponent_dies_to_no_hit(self):
        """Deathtouch destroys and indestructible stops it (702.12b), so the
        death the deathtouch and strike-first rows read cannot happen."""
        state, carrier = _attacker_and_blockers(
            attacker=(1, 2), blockers=((1, 3),), attacker_keywords=("deathtouch",),
            blocker_keywords=("indestructible",))
        assert not _qualifies("deathtouch", state, carrier)
        state, carrier = _attacker_and_blockers(
            attacker=(3, 2), blockers=((1, 3),), attacker_keywords=("first_strike",),
            blocker_keywords=("indestructible",), substep="first_strike")
        assert _observable("first_strike", state, carrier) == ["opponent.damage_taken"]
