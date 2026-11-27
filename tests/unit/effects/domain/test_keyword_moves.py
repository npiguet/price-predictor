"""The right move of each gate-2 effect, from the combat record's own events.

The fixture's two qualifying combats are real: a first striker's hit that
kills its lone blocker, and a trampler's excess to the defending player. The
other effects occur in no committed record (the engine boards that show them
live in the connector's Java harnesses, which write no Python fixture), so
their boards are the predicate tests' own hand-built ones, with the step's
events written in the collector's spelling (a ``damage_dealt`` naming its
``source``, a ``zone_change`` from the battlefield). Each hand-built board
is checked to qualify its effect, so it is a combat gate 2 would score.
"""

from __future__ import annotations

import pytest

from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    KeywordResolver,
    combat_participant,
    qualifying_observations,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.keyword_moves import damage_dealt_by, right_move
from effects.domain.records import CombatPayload, RecordKind
from tests.unit.effects.builders import GEN3_RECORDS, GEN3_SIDECAR_FOLDERS
from tests.unit.effects.domain.test_damage_step_predicates import (
    BlockedAttack,
    _attacker_and_blockers,
    _unblocked_attacker,
)
from tests.unit.effects.record_builders import make_record

#: The fixture's real first-strike combat: E220, a 6-power first striker, kills
#: its lone blocker E63 with 6 ``normal`` damage.
_FIRST_STRIKE = "3550376f-3c30-4231-90cd-cc0f1b308e87.0-mwqvqdyrb9c0.602"
#: The fixture's real trample combat: E43, 6 power, assigns 3 to its blocker E9
#: and 3 to the defending player P0.
_TRAMPLE = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv.974"


@pytest.fixture(scope="module")
def fixture_observations():
    """``(record id, keyword) -> (record, participant)`` for the fixture's
    qualifying real combats."""
    from effects.infrastructure.converted_trees import SourceTrees
    from effects.infrastructure.shard_io import read_shard

    resolver = KeywordResolver(SourceTrees.of(list(GEN3_SIDECAR_FOLDERS)).sidecar_cache())
    return {
        (record.record_id, row.keyword): (record, participant)
        for record in read_shard(GEN3_RECORDS)
        if not record.fork and not record.interventional
        for row, participant in qualifying_observations(record, DAMAGE_STEP_KEYWORDS, resolver)
    }


def _effect(keyword: str, label: str):
    return next(e for e in KEYWORDS_BY_NAME[keyword].effects if e.label == label)


def _moves(record, participant, keyword: str) -> dict[str, list[float]]:
    """Every observable effect's right move at each of its subjects."""
    return {
        effect.label: [right_move(effect, participant, subject, record.payload.events)
                       for subject in participant.subjects(effect.subject)]
        for effect in KEYWORDS_BY_NAME[keyword].effects if effect.observable(participant)
    }


class TestTheFixturesRealCombats:
    def test_a_first_strikers_hit_moves_damage_by_its_amount_and_the_death_by_one(
            self, fixture_observations):
        record, participant = fixture_observations[(_FIRST_STRIKE, "first_strike")]
        assert _moves(record, participant, "first_strike") == {
            "opponent.damage_taken": [6.0], "opponent.zone_outcome/died": [1.0],
        }

    def test_trample_moves_the_defending_players_life_by_the_excess_dealt(
            self, fixture_observations):
        record, participant = fixture_observations[(_TRAMPLE, "trample")]
        assert _moves(record, participant, "trample") == {
            "defending_player.life_delta": [-3.0],
        }


# ── hand-built boards ───────────────────────────────────────────────────


def _hit(target: str, amount: int, *, source: str = "A", damage_type: str | None = "normal"):
    params = {"amount": amount, "combat": True, "source": source}
    if damage_type is not None:
        params["damage_type"] = damage_type
    return Event(EventType.DAMAGE_DEALT, (target,), params)


def _death(entity: str) -> Event:
    return Event(EventType.ZONE_CHANGE, (entity,),
                 {"from_zone": "battlefield", "to_zone": "graveyard"})


def _combat(state, *events, assignment=None):
    payload = CombatPayload(assignment_choices=assignment or {}, events=tuple(events))
    return make_record(RecordKind.COMBAT, state=state, payload=payload)


def _overlay_keywords(entity) -> set[str]:
    return set(entity.granted_temporary.keywords)


def _right(record, carrier: str, keyword: str, label: str) -> list[float]:
    """``label``'s right move at each subject, the board checked to qualify it."""
    participant = combat_participant(record, record.state.entity(carrier), _overlay_keywords)
    effect = _effect(keyword, label)
    assert effect.observable(participant), f"{keyword}: the board does not qualify {label}"
    return [right_move(effect, participant, subject, record.payload.events)
            for subject in participant.subjects(effect.subject)]


class TestFirstAndDoubleStrike:
    """In the first-strike step's record the carrier's hit lands only with the keyword."""

    @pytest.mark.parametrize("keyword", ["first_strike", "double_strike"])
    def test_an_unblocked_strike_moves_the_defending_players_life_down_by_its_damage(
            self, keyword):
        state, _ = _unblocked_attacker(3, 3, substep="first_strike", keywords=(keyword,))
        record = _combat(state, _hit("P1", 3, damage_type=None))
        assert _right(record, "A", keyword, "defending_player.life_delta") == [-3.0]

    def test_a_double_strikers_first_hit_moves_the_blockers_damage_and_death(self):
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(4, 4), blockers=((2, 3),), attacker_keywords=("double_strike",),
            substep="first_strike"))
        record = _combat(state, _hit("B0", 4), _death("B0"))
        assert _right(record, "A", "double_strike", "opponent.damage_taken") == [4.0]
        assert _right(record, "A", "double_strike", "opponent.zone_outcome/died") == [1.0]


class TestDeathtouch:
    def test_the_opponent_s_death_moves_by_one(self):
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 1), blockers=((3, 3),), attacker_keywords=("deathtouch",)))
        record = _combat(state, _hit("B0", 1, damage_type="deathtouch"), _death("B0"))
        assert _right(record, "A", "deathtouch", "opponent.zone_outcome/died") == [1.0]

    def test_an_observed_survival_against_the_rule_is_a_right_move_of_zero(self):
        """The rule says the deathtouch hit kills, the record shows the blocker
        alive (a regeneration shield, say): nothing shows the twin to differ."""
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 1), blockers=((3, 3),), attacker_keywords=("deathtouch",)))
        record = _combat(state, _hit("B0", 1, damage_type="deathtouch"))
        assert _right(record, "A", "deathtouch", "opponent.zone_outcome/died") == [0.0]


class TestLifelink:
    def test_the_controller_gains_everything_the_carrier_dealt(self):
        """702.15b: a trampling lifelinker's damage to its blocker and to the
        player both gain life; another creature's damage gains none."""
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(5, 5), blockers=((2, 2),), attacker_keywords=("lifelink", "trample")))
        record = _combat(state, _hit("B0", 2), _hit("P1", 3, damage_type=None),
                         _hit("A", 2, source="B0"), assignment={"A": {"B0": 2, "P1": 3}})
        assert _right(record, "A", "lifelink", "controller.life_delta") == [5.0]

    def test_a_prevented_hit_is_a_right_move_of_zero(self):
        """A Fog prevents the damage: the record holds a ``damage_prevented``
        and no ``damage_dealt``, so lifelink gains nothing to remove."""
        state, _ = _unblocked_attacker(3, 3, keywords=("lifelink",))
        prevented = Event(EventType.DAMAGE_PREVENTED, ("P1",), {"amount": 3, "source": "A"})
        record = _combat(state, prevented)
        assert _right(record, "A", "lifelink", "controller.life_delta") == [0.0]


class TestIndestructible:
    def _blocked_into_lethal(self, *events):
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((3, 3),), attacker_keywords=("indestructible",)))
        return _combat(state, _hit("B0", 2), _hit("A", 3, source="B0"), *events)

    def test_a_carrier_that_lived_through_lethal_damage_moves_its_death_down_by_one(self):
        record = self._blocked_into_lethal()
        assert _right(record, "A", "indestructible", "carrier.zone_outcome/died") == [-1.0]

    def test_a_carrier_that_died_anyway_is_a_right_move_of_zero(self):
        record = self._blocked_into_lethal(_death("A"))
        assert _right(record, "A", "indestructible", "carrier.zone_outcome/died") == [0.0]


class TestWitherAndInfectOnCreatures:
    """-1/-1 counters instead of marked damage, read at each creature dealt damage."""

    @pytest.mark.parametrize("keyword", ["wither", "infect"])
    def test_each_blocker_dealt_damage_moves_by_what_it_was_dealt(self, keyword):
        state, _ = _attacker_and_blockers(BlockedAttack(
            attacker=(3, 3), blockers=((2, 2), (2, 2)), attacker_keywords=(keyword,)))
        record = _combat(state, _hit("B0", 2, damage_type="m1m1_counters"),
                         _hit("B1", 1, damage_type="m1m1_counters"),
                         assignment={"A": {"B0": 2, "B1": 1}})
        assert _right(record, "A", keyword, "opponent.counters_delta_m1m1") == [2.0, 1.0]
        assert _right(record, "A", keyword, "opponent.damage_taken") == [-2.0, -1.0]


class TestInfectOnThePlayer:
    def test_poison_moves_up_and_life_stays_up_by_the_damage_dealt(self):
        """702.90b: without infect the player would have lost the life."""
        state, _ = _unblocked_attacker(2, 2, keywords=("infect",))
        record = _combat(state, _hit("P1", 2, damage_type=None))
        assert _right(record, "A", "infect", "defending_player.poison_delta") == [2.0]
        assert _right(record, "A", "infect", "defending_player.life_delta") == [2.0]


def test_damage_dealt_by_reads_the_source_and_the_subject():
    events = (_hit("B0", 2), _hit("P1", 3, damage_type=None), _hit("B0", 4, source="C"))
    assert damage_dealt_by("A", events) == 5.0
    assert damage_dealt_by("A", events, "B0") == 2.0
    assert damage_dealt_by("C", events, "P1") == 0.0
