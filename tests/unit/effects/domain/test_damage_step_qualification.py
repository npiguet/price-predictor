"""Gate 2's table qualifies only what the record can show.

One case per board the review found the table counting against agreement:
a blocker carrier in a multi-block (deathtouch 702.2, indestructible
702.12), wither and infect read at the first blocker instead of the ones the
carrier's damage was dealt to (702.80, 702.90), life and damage effects on a
carrier whose damage is poison or counters, a first striker into a blocker
protected from it (702.7, 702.16e), and lifelink on an attacker whose
defender is gone (506.4c). The boards are the predicate tests' own, edited to
the case; the step's damage assignment is the record's
``assignment_choices``, ``source -> {target: damage}``.
"""

from __future__ import annotations

import dataclasses

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    KeywordResolver,
    qualifying_observations,
)
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.rule_detectors.keyword_combat import KeywordCombatMatcher
from effects.domain.state_snapshot import CombatStatus, GrantedTemporary
from tests.unit.effects.damage_step_boards import CreatureTraits, creature
from tests.unit.effects.domain.test_damage_step_predicates import (
    BlockedAttack,
    _attacker_and_blockers,
    _board,
    _planeswalker,
    _unblocked_attacker,
)
from tests.unit.effects.record_builders import make_record


def _record(state, assignment=None):
    payload = CombatPayload(assignment_choices=assignment or {})
    return make_record(RecordKind.COMBAT, state=state, payload=payload)


def _carriers(keyword: str, record) -> list[str]:
    """The carriers ``keyword``'s row qualifies in ``record``."""
    found = qualifying_observations(record, (KEYWORDS_BY_NAME[keyword],), KeywordResolver())
    return [participant.carrier.id for _row, participant in found]


def _with_keywords(state, entity_id: str, *keywords: str):
    """``state`` with ``entity_id`` given ``keywords`` alone."""
    entities = tuple(
        dataclasses.replace(e, granted_temporary=GrantedTemporary(keywords=keywords))
        if e.id == entity_id else e for e in state.entities)
    return dataclasses.replace(state, entities=entities)


def _double_block(attacker: tuple[int, int], blockers, keyword: str):
    """``A`` blocked by ``B0`` and ``B1``, ``B0`` carrying ``keyword``."""
    state, _attacker = _attacker_and_blockers(BlockedAttack(attacker=attacker, blockers=blockers))
    return _with_keywords(state, "B0", keyword)


class TestABlockerCarrierInAMultiBlock:
    def test_a_deathtouch_blocker_in_a_double_block_qualifies_only_if_the_attacker_would_live(self):
        """Every blocker deals its damage to the attacker (510.1c): a 3/3
        blocked by a 1/1 and a 2/2 dies without deathtouch (702.2b)."""
        dies_anyway = _double_block((3, 3), ((1, 1), (2, 2)), "deathtouch")
        assert _carriers("deathtouch", _record(dies_anyway)) == []
        lives = _double_block((4, 4), ((1, 1), (2, 2)), "deathtouch")
        assert _carriers("deathtouch", _record(lives)) == ["B0"]

    def test_an_indestructible_blocker_reads_the_damage_the_attacker_assigned_it(self):
        """The attacker splits its damage by its controller's choice
        (510.1c); 2 of 4 to a 2/3 is no lethal damage (702.12b)."""
        state = _double_block((4, 4), ((2, 3), (2, 2)), "indestructible")
        assert _carriers("indestructible", _record(state, {"A": {"B0": 2, "B1": 2}})) == []
        assert _carriers("indestructible", _record(state, {"A": {"B0": 3, "B1": 1}})) == ["B0"]


class TestWitherAndInfectAtEachCreatureDealtDamage:
    def _infect_double_block(self):
        state, _attacker = _attacker_and_blockers(BlockedAttack(
            attacker=(3, 3), blockers=((2, 2), (2, 2)), attacker_keywords=("infect",)))
        return state

    def _outcome(self, record) -> tuple[str, ...]:
        return KeywordCombatMatcher.of({"keyword": "infect"})(record, None).outcome

    def test_infect_with_two_blockers_where_the_first_is_assigned_nothing(self):
        """702.90b: the counters land where the damage was dealt."""
        record = _record(self._infect_double_block(), {"A": {"B1": 3}})
        assert self._outcome(record) == ("B1",)

    def test_infect_split_between_two_blockers_is_read_at_both(self):
        record = _record(self._infect_double_block(), {"A": {"B0": 1, "B1": 2}})
        assert self._outcome(record) == ("B0", "B1")

    def test_infect_assigned_to_no_blocker_qualifies_nothing(self):
        assert _carriers("infect", _record(self._infect_double_block())) == []


class TestPoisonAndCountersAreNoLifeOrDamage:
    def test_an_unblocked_infect_first_striker_moves_no_life(self):
        """702.90b: its damage to a player is poison counters, not life loss."""
        state, _carrier = _unblocked_attacker(
            keywords=("first_strike", "infect"), substep="first_strike")
        assert _carriers("first_strike", _record(state)) == []

    def test_an_infect_trampler_moves_no_life(self):
        state, _carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(5, 5), blockers=((1, 1),), attacker_keywords=("trample", "infect")))
        assert _carriers("trample", _record(state)) == []

    def test_a_wither_or_infect_first_striker_marks_no_damage(self):
        """702.80a, 702.90a: its damage to a creature is -1/-1 counters."""
        for keyword in ("wither", "infect"):
            state, _carrier = _attacker_and_blockers(BlockedAttack(
                attacker=(2, 2), blockers=((1, 3),),
                attacker_keywords=("first_strike", keyword), substep="first_strike"))
            assert _carriers("first_strike", _record(state)) == [], keyword


def test_a_first_striker_into_a_blocker_protected_from_it_qualifies_nothing():
    """702.16e: the protection prevents the damage first strike would move."""
    state, carrier = _attacker_and_blockers(BlockedAttack(
        attacker=(2, 2), blockers=((1, 3),), attacker_keywords=("first_strike",),
        blocker_keywords=("protection_from_red",), substep="first_strike"))
    red = dataclasses.replace(carrier, colors=("R",))
    assert _carriers("first_strike", _record(
        dataclasses.replace(state, entities=(red, *state.entities[1:])))) == []
    green = dataclasses.replace(carrier, colors=("G",))
    assert _carriers("first_strike", _record(
        dataclasses.replace(state, entities=(green, *state.entities[1:])))) == ["A"]


def test_lifelink_with_the_defender_gone_qualifies_nothing():
    """506.4c: an attacker whose planeswalker left attacks ``E-1`` and deals
    no damage; one whose planeswalker is still there does."""
    gone = creature("A", 3, 3, CreatureTraits(
        keywords=("lifelink",), combat=CombatStatus(attacking="E-1")))
    assert _carriers("lifelink", _record(_board(gone))) == []
    there = dataclasses.replace(gone, combat=CombatStatus(attacking="E9"))
    assert _carriers("lifelink", _record(_board(there, _planeswalker()))) == ["A"]
