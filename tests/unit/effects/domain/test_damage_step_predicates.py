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

from dataclasses import dataclass

import pytest

from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    combat_participant,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import RecordKind
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    PlayerState,
    StateSnapshot,
)
from tests.unit.effects.damage_step_boards import (
    CreatureTraits,
    creature,
    damage_step_global,
)
from tests.unit.effects.file_builders import fresh_interpreter_env
from tests.unit.effects.record_builders import make_entity, make_record, make_snapshot

_WHITE_KNIGHT = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "static", 0)

#: ``CombatStatus.attacking`` for an attack on a planeswalker: its entity ref.
_PLANESWALKER_DEFENDER = "E9"

#: ``CombatStatus.attacking`` for an attack on a player: the player's ref.
_PLAYER_DEFENDER = "P1"


def _board(*entities: EntityState, substep: str = "regular") -> StateSnapshot:
    return make_snapshot(
        global_=damage_step_global(substep),
        players=(
            PlayerState(id="P0", life=20, hand=3, library=30, graveyard=2),
            PlayerState(id="P1", life=17, hand=4, library=28, graveyard=1),
        ),
        entities=entities,
    )


@dataclass(frozen=True)
class BlockedAttack:
    """``A``'s attack on ``P1``, blocked by one creature ``B<i>`` per entry of
    ``blockers``; each size is ``(power, toughness)``."""

    attacker: tuple[int, int] = (2, 2)
    blockers: tuple[tuple[int, int], ...] = ((2, 2),)
    attacker_damage: int = 0
    attacker_keywords: tuple[str, ...] = ()
    blocker_damage: int = 0
    blocker_keywords: tuple[str, ...] = ()
    substep: str = "regular"


def _attacker_and_blockers(
    attack: BlockedAttack = BlockedAttack(),
) -> tuple[StateSnapshot, EntityState]:
    blocker_ids = tuple(f"B{i}" for i in range(len(attack.blockers)))
    carrier = creature("A", *attack.attacker, CreatureTraits(
        damage=attack.attacker_damage, keywords=attack.attacker_keywords,
        combat=CombatStatus(
            attacking=_PLAYER_DEFENDER, blocked_by=blocker_ids,
            became_blocked=bool(blocker_ids),
        ),
    ))
    blocking = tuple(
        creature(blocker_id, *stats, CreatureTraits(
            controller="P1", damage=attack.blocker_damage,
            keywords=attack.blocker_keywords, combat=CombatStatus(blocking=("A",)),
        ))
        for blocker_id, stats in zip(blocker_ids, attack.blockers)
    )
    return _board(carrier, *blocking, substep=attack.substep), carrier


def _planeswalker() -> EntityState:
    """A defender that is not a player."""
    return make_entity(
        "E9", "Chandra, Torch of Defiance", controller="P1", types=("planeswalker",),
    )


def _unblocked_attacker(
    power: int = 3, toughness: int = 3, *, substep: str = "regular",
    keywords: tuple[str, ...] = (),
):
    carrier = creature("A", power, toughness, CreatureTraits(
        keywords=keywords, combat=CombatStatus(attacking=_PLAYER_DEFENDER),
    ))
    return _board(carrier, substep=substep), carrier


def _overlay_keywords(entity) -> set[str]:
    return set(entity.granted_temporary.keywords)


def _participant_of(state, carrier, keywords_of=_overlay_keywords):
    """The carrier's participant in a combat record of ``state``."""
    return combat_participant(_combat_record(state), carrier, keywords_of)


def _qualifies(keyword: str, state, carrier, keywords_of=_overlay_keywords) -> bool:
    participant = _participant_of(state, carrier, keywords_of)
    assert participant is not None
    return KEYWORDS_BY_NAME[keyword].qualifies(participant)


def _observable(
    keyword: str, state, carrier, keywords_of=_overlay_keywords,
) -> list[str]:
    """The labels of the effects this combat can actually show."""
    participant = _participant_of(state, carrier, keywords_of)
    assert participant is not None
    return [
        effect.label for effect in KEYWORDS_BY_NAME[keyword].effects
        if effect.observable(participant)
    ]



def _combat_record(state: StateSnapshot):
    return make_record(RecordKind.COMBAT, state=state)



class TestDealsDamageAtAll:
    """A creature with no power deals none, whatever else is true of it."""

    @pytest.mark.parametrize(
        "keyword",
        ["first_strike", "double_strike", "deathtouch", "lifelink", "wither",
         "infect"],
    )
    @pytest.mark.parametrize("substep", ["first_strike", "regular"])
    def test_a_powerless_carrier_qualifies_for_nothing(self, keyword, substep):
        carrier = creature(
            "A", 0, 3, CreatureTraits(
                keywords=(keyword,),
                combat=CombatStatus(attacking=_PLAYER_DEFENDER, blocked_by=("B0",),
                                    became_blocked=True),
            ),
        )
        blocker = creature(
            "B0", 1, 1, CreatureTraits(controller="P1", combat=CombatStatus(blocking=("A",))),
        )
        state = _board(carrier, blocker, substep=substep)
        assert not _qualifies(keyword, state, carrier)

    def test_a_powerless_carrier_can_still_be_indestructible(self):
        """Indestructible is about damage coming in, not damage going out."""
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(0, 2), blockers=((3, 3),),
        ))
        assert _qualifies("indestructible", state, carrier)



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

    def test_a_blocker_whose_attacker_is_gone_deals_nothing(self):
        carrier = creature(
            "B0", 2, 2, CreatureTraits(controller="P1", combat=CombatStatus(blocking=("A",))),
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
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((1, 3),),
        ))
        assert _qualifies("deathtouch", state, carrier)

    def test_an_opponent_that_would_die_anyway_does_not(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(3, 3), blockers=((1, 3),),
        ))
        assert not _qualifies("deathtouch", state, carrier)

    def test_damage_already_marked_counts_against_the_opponents_toughness(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((1, 3),), blocker_damage=2,
        ))
        assert not _qualifies("deathtouch", state, carrier)

    def test_several_blockers_make_the_assignment_unpredictable(self):
        """Which blocker the two damage goes to is Forge's choice, not a rule."""
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((1, 1), (1, 5)),
        ))
        assert not _qualifies("deathtouch", state, carrier)


class TestTrample:
    def test_a_blocked_attacker_bigger_than_its_blockers_qualifies(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(4, 4), blockers=((1, 3),),
        ))
        assert _qualifies("trample", state, carrier)

    def test_blockers_that_soak_every_point_do_not_qualify(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(4, 4), blockers=((1, 2), (1, 2)),
        ))
        assert not _qualifies("trample", state, carrier)

    def test_damage_already_on_a_blocker_leaves_less_to_soak(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(4, 4), blockers=((1, 4),), blocker_damage=1,
        ))
        assert _qualifies("trample", state, carrier)

    def test_an_unblocked_attacker_does_not_qualify(self):
        """Nothing is in the way, so there is no excess to run over."""
        state, carrier = _unblocked_attacker(power=4)
        assert not _qualifies("trample", state, carrier)

    def test_a_blocker_does_not_qualify(self):
        state, _ = _attacker_and_blockers(BlockedAttack(attacker=(1, 1), blockers=((4, 4),)))
        assert not _qualifies("trample", state, state.entity("B0"))


class TestIndestructible:
    def test_lethal_incoming_damage_qualifies(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 3), blockers=((3, 3),),
        ))
        assert _qualifies("indestructible", state, carrier)

    def test_survivable_incoming_damage_does_not(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 4), blockers=((3, 3),),
        ))
        assert not _qualifies("indestructible", state, carrier)

    def test_damage_already_marked_makes_survivable_damage_lethal(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 4), blockers=((3, 3),), attacker_damage=1,
        ))
        assert _qualifies("indestructible", state, carrier)

    def test_several_blockers_add_up(self):
        """A sum does not depend on how Forge splits the damage, so unlike the
        first-strike rules this one still reads a multi-block."""
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(1, 4), blockers=((2, 2), (2, 2)),
        ))
        assert _qualifies("indestructible", state, carrier)

    def test_a_deathtouch_blocker_makes_any_damage_lethal(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((1, 1),),
            blocker_keywords=("deathtouch",),
        ))
        assert _qualifies("indestructible", state, carrier)

    def test_a_powerless_deathtouch_blocker_deals_no_lethal_damage(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((0, 1),),
            blocker_keywords=("deathtouch",),
        ))
        assert not _qualifies("indestructible", state, carrier)

    def test_an_unblocked_attacker_takes_nothing(self):
        state, carrier = _unblocked_attacker()
        assert not _qualifies("indestructible", state, carrier)

    def test_an_infect_blocker_kills_through_indestructible(self):
        """Infect and wither damage lands as -1/-1 counters, which shrink the
        carrier to death without destroying it, so the keyword saves nothing."""
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((2, 2),),
            blocker_keywords=("infect",),
        ))
        assert not _qualifies("indestructible", state, carrier)

    def test_a_wither_blocker_kills_through_indestructible(self):
        state, carrier = _attacker_and_blockers(BlockedAttack(
            attacker=(2, 2), blockers=((2, 2),),
            blocker_keywords=("wither",),
        ))
        assert not _qualifies("indestructible", state, carrier)


class TestTheRowsFields:
    def test_a_row_names_each_head_field_once(self):
        """Infect reads two fields on the creature and two on the player."""
        assert KEYWORDS_BY_NAME["infect"].fields == (
            "counters_delta_m1m1", "damage_taken", "poison_delta", "life_delta",
        )



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
