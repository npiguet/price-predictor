"""Who a carrier fights in a combat record, and which carriers a record holds.

``combat_participant`` reads, for one creature in combat, the creatures it
faces and the player its attack is aimed at; ``qualifying_observations`` finds
every carrier of a record gate 2 can observe, its keywords read from the
overlay and the sidecar alike (FR-120, FR-122).
"""

from __future__ import annotations

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    KeywordResolver,
    Subject,
    qualifying_observations,
)
from effects.domain.provenance import SidecarLine
from effects.domain.state_snapshot import (
    CombatStatus,
)
from tests.unit.effects.damage_step_boards import CreatureTraits, creature
from tests.unit.effects.domain.test_damage_step_predicates import (
    _PLANESWALKER_DEFENDER,
    _PLAYER_DEFENDER,
    _WHITE_KNIGHT,
    BlockedAttack,
    _attacker_and_blockers,
    _board,
    _combat_record,
    _participant_of,
    _planeswalker,
    _qualifies,
    _unblocked_attacker,
)


class TestCombatParticipant:
    """Who the carrier is fighting, and which player the attack is aimed at."""

    def test_an_attacker_faces_every_creature_blocking_it(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(blockers=((1, 1), (2, 2))))
        participant = _participant_of(state, carrier)
        assert [o.id for o in participant.opponents] == ["B0", "B1"]

    def test_an_attackers_controller_and_defending_player_are_the_two_sides(self):
        state, carrier = _attacker_and_blockers()
        participant = _participant_of(state, carrier)
        assert participant.carrier.controller == "P0"
        assert participant.defending_player == "P1"

    def test_an_unblocked_attacker_has_no_opponent_but_still_has_a_defender(self):
        state, carrier = _unblocked_attacker()
        participant = _participant_of(state, carrier)
        assert participant.opponents == ()
        assert participant.defending_player == "P1"

    def test_a_blocker_faces_the_attacker_it_blocks(self):
        state, _ = _attacker_and_blockers()
        blocker = state.entity("B0")
        participant = _participant_of(state, blocker)
        assert [o.id for o in participant.opponents] == ["A"]
        assert participant.carrier.controller == "P1"

    def test_a_blocker_has_no_defending_player(self):
        """Its damage lands on the attacker, never on a player."""
        state, _ = _attacker_and_blockers()
        participant = _participant_of(state, state.entity("B0"))
        assert participant.defending_player is None

    def test_it_carries_the_damage_step_the_record_covers(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(substep="first_strike"))
        assert _participant_of(state, carrier).combat_substep == "first_strike"

    def test_the_defending_player_is_the_ref_the_attack_names(self):
        """The defender's ref is the defending player as it is."""
        carrier = creature(
            "A", 2, 2, CreatureTraits(combat=CombatStatus(attacking="P1")),
        )
        participant = _participant_of(_board(carrier), carrier)
        assert participant.defending_player == "P1"

    def test_a_creature_outside_combat_is_no_participant(self):
        carrier = creature("A", 2, 2)
        assert _participant_of(_board(carrier), carrier) is None

    def test_a_blocker_whose_attacker_left_the_board_has_no_opponent(self):
        carrier = creature(
            "B0", 2, 2, CreatureTraits(controller="P1", combat=CombatStatus(blocking=("A",))),
        )
        participant = _participant_of(_board(carrier), carrier)
        assert participant.opponents == ()

    def test_an_attack_on_a_planeswalker_has_no_defending_player(self):
        """Its damage lands on the planeswalker, not on that player's life."""
        carrier = creature(
            "A", 4, 4, CreatureTraits(combat=CombatStatus(attacking=_PLANESWALKER_DEFENDER)),
        )
        participant = _participant_of(
            _board(carrier, _planeswalker()), carrier,
        )
        assert participant.defending_player is None

    def test_an_attack_on_a_battle_off_the_snapshot_has_no_defending_player(self):
        """A card ref names no player, whether or not the board holds it."""
        carrier = creature(
            "A", 4, 4, CreatureTraits(combat=CombatStatus(attacking="E51")),
        )
        participant = _participant_of(_board(carrier), carrier)
        assert participant.defending_player is None

    def test_an_attack_on_a_departed_planeswalker_has_no_defending_player(self):
        """The placeholder a departed defender leaves (506.4c) is ``E-1``."""
        carrier = creature(
            "A", 4, 4, CreatureTraits(combat=CombatStatus(attacking="E-1")),
        )
        participant = _participant_of(_board(carrier), carrier)
        assert participant.defending_player is None

    def test_trample_over_a_planeswalkers_blockers_is_not_an_observation(self):
        carrier = creature(
            "A", 4, 4,
            CreatureTraits(combat=CombatStatus(
                attacking=_PLANESWALKER_DEFENDER, blocked_by=("B0",),
                became_blocked=True,
            )),
        )
        blocker = creature(
            "B0", 1, 1, CreatureTraits(controller="P1", combat=CombatStatus(blocking=("A",))),
        )
        state = _board(carrier, _planeswalker(), blocker)
        assert not _qualifies("trample", state, carrier)


# ── who is observed: a record's carriers, through both keyword channels ──


class TestSubjectResolution:
    """Where each effect's subject lands on one blocked attack."""

    @pytest.mark.parametrize(("subject", "expected"), [
        (Subject.CARRIER, "A"),
        (Subject.OPPONENT, "B0"),
        (Subject.CONTROLLER, "P0"),
        (Subject.DEFENDING_PLAYER, "P1"),
    ])
    def test_a_blocked_attackers_subjects(self, subject, expected):
        state, carrier = _attacker_and_blockers()
        assert _participant_of(state, carrier).subjects(subject) == (expected,)


#: White Knight's printed first strike, as its sidecar line spells it.
_WHITE_KNIGHT_LINE = SidecarLine(
    line_index=0, line_kind="static", provenance=(_WHITE_KNIGHT,),
    script_text="First Strike", prose_offset=0,
)


class _WhiteKnightSidecar:
    """A sidecar lookup that answers for White Knight's first-strike line alone."""

    def line_or_none(self, key):
        return _WHITE_KNIGHT_LINE if key == _WHITE_KNIGHT else None



class TestQualifyingObservations:
    """``qualifying_observations``: every carrier of a combat record the gate
    can observe, its keywords read from the overlay and the sidecar alike."""

    _FIRST_STRIKE = (KEYWORDS_BY_NAME["first_strike"],)

    def test_a_printed_keyword_is_found_through_the_sidecar(self):
        carrier = creature(
            "A", 2, 2, CreatureTraits(
                printed=(_WHITE_KNIGHT,), combat=CombatStatus(attacking=_PLAYER_DEFENDER),
            ),
        )
        record = _combat_record(_board(carrier, substep="first_strike"))
        found = qualifying_observations(
            record, self._FIRST_STRIKE, KeywordResolver(_WhiteKnightSidecar()),
        )
        assert [(row.keyword, p.carrier.id) for row, p in found] == [
            ("first_strike", "A"),
        ]

    def test_an_overlay_keyword_is_found_without_a_sidecar(self):
        state, _carrier = _unblocked_attacker(
            keywords=("first_strike",), substep="first_strike",
        )
        found = qualifying_observations(
            _combat_record(state), self._FIRST_STRIKE, KeywordResolver(None),
        )
        assert len(found) == 1

    def test_two_carriers_in_one_record_are_two_observations(self):
        state, _carrier = _attacker_and_blockers(BlockedAttack(
            attacker_keywords=("lifelink",), blocker_keywords=("lifelink",),
        ))
        found = qualifying_observations(
            _combat_record(state), (KEYWORDS_BY_NAME["lifelink"],), KeywordResolver(None),
        )
        assert sorted(p.carrier.id for _row, p in found) == ["A", "B0"]
