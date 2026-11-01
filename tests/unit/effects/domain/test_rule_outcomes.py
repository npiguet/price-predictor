"""``rule_outcomes``: each rule function on the cases its former home pinned (T012)."""

from __future__ import annotations

import pytest

from effects.domain.rule_outcomes import (
    EffectKind,
    damage_lethal,
    lethal,
    may_block,
    shrink_lethal,
    step_deaths,
    strikes_in_step,
)
from effects.domain.state_snapshot import EntityState, PowerToughness
from tests.unit.effects.builders import make_entity


def _creature(toughness: int, *, damage: int = 0) -> EntityState:
    return make_entity("E1", "E1", controller="P1", types=("creature",),
                       pt=PowerToughness(base=(2, toughness)), damage=damage)


class TestLethal:
    def test_damage_kills_at_toughness_counting_marked_damage(self):
        assert lethal(EffectKind.DAMAGE, 3, _creature(3))
        assert not lethal(EffectKind.DAMAGE, 2, _creature(3))
        assert lethal(EffectKind.DAMAGE, 2, _creature(3, damage=1))

    def test_a_shrink_kills_at_zero_toughness_or_at_marked_damage(self):
        assert lethal(EffectKind.SHRINK, 2, _creature(2))
        assert not lethal(EffectKind.SHRINK, 2, _creature(3))
        assert lethal(EffectKind.SHRINK, 2, _creature(3, damage=1))

    def test_zero_toughness_dies_to_no_damage(self):
        assert damage_lethal(0, 0, 0)
        assert not damage_lethal(1, 0, 0)
        assert shrink_lethal(1, 0, 1)


class TestStepDeaths:
    @pytest.mark.parametrize("attacker, blocker, keywords, step, expected", [
        ((3, 3, 0), (2, 2, 0), ({}, {}), "regular", (False, True)),
        ((2, 2, 0), (2, 3, 0), ({}, {}), "regular", (True, False)),
        ((2, 2, 0), (1, 3, 1), ({}, {}), "regular", (False, True)),
        ((1, 5, 0), (1, 5, 0), ({"deathtouch"}, {}), "regular", (False, True)),
        ((5, 5, 0), (5, 5, 0), ({}, {"indestructible"}), "regular", (True, False)),
        ((3, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, True)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, False)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "regular", (True, False)),
        ((5, 3, 0), (3, 3, 0), ({}, {"first_strike"}), "first_strike", (True, False)),
        ((2, 2, 0), (1, 4, 0), ({"double_strike"}, {}), "first_strike", (False, False)),
        ((2, 2, 0), (1, 4, 2), ({"double_strike"}, {}), "regular", (False, True)),
        ((2, 2, 0), (0, 4, 0), ({}, {}), "first_strike", (False, False)),
    ])
    def test_the_rule(self, attacker, blocker, keywords, step, expected):
        assert step_deaths(
            attacker, blocker, frozenset(keywords[0]), frozenset(keywords[1]), step) == expected

    @pytest.mark.parametrize("keywords, step, strikes", [
        (set(), "regular", True),
        (set(), "first_strike", False),
        ({"first_strike"}, "first_strike", True),
        ({"first_strike"}, "regular", False),
        ({"double_strike"}, "first_strike", True),
        ({"double_strike"}, "regular", True),
    ])
    def test_who_strikes_in_which_step(self, keywords, step, strikes):
        assert strikes_in_step(keywords, step) is strikes


class TestMayBlock:
    @pytest.mark.parametrize("attacker, blocker, colours, types, expected", [
        ((), (), ("G",), ("creature",), True),
        (("flying",), (), ("G",), ("creature",), False),
        (("flying",), ("reach",), ("G",), ("creature",), True),
        (("horsemanship",), ("flying",), ("G",), ("creature",), False),
        (("fear",), (), ("B",), ("creature",), True),
        (("fear",), (), ("G",), ("artifact", "creature"), True),
        (("fear",), (), ("G",), ("creature",), False),
        (("intimidate",), (), ("R",), ("creature",), True),
        (("intimidate",), (), ("G",), ("creature",), False),
        (("shadow",), (), ("G",), ("creature",), False),
        ((), ("shadow",), ("G",), ("creature",), False),
        (("unblockable",), ("reach",), ("G",), ("creature",), False),
        (("protection_from_green",), (), ("G",), ("creature",), False),
        (("protection_from_blue",), (), ("G",), ("creature",), True),
    ])
    def test_the_rule(self, attacker, blocker, colours, types, expected):
        assert may_block(attacker, ("R",), blocker, colours, types) is expected
