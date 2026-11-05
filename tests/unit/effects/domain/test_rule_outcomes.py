"""``rule_outcomes``: each rule function on the cases its former home pinned (T012),
and protection's rare qualities (702.16, T137)."""

from __future__ import annotations

import pytest

from effects.domain.rule_outcomes import (
    Characteristics,
    EffectKind,
    ProtectionQuality,
    QualityKind,
    damage_lethal,
    damage_prevented,
    lethal,
    may_block,
    may_enchant,
    may_target,
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

    @pytest.mark.parametrize("protection, blocker, expected", [
        (ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin"), {"subtypes": ("goblin",)}, False),
        (ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin"), {"subtypes": ("elf",)}, True),
        (ProtectionQuality(QualityKind.CARD_TYPE, "artifact"),
         {"types": ("artifact", "creature")}, False),
        (ProtectionQuality(QualityKind.CONDITION, "multicolored"), {"colours": ("W", "U")}, False),
        (ProtectionQuality(QualityKind.CONDITION, "multicolored"), {"colours": ("W",)}, True),
        (ProtectionQuality(QualityKind.CONDITION, "mana value 3 or greater"),
         {"mana_value": 3}, False),
        (ProtectionQuality.everything(), {}, False),
    ])
    def test_a_rare_protection_forbids_the_blockers_with_its_quality(
            self, protection, blocker, expected):
        assert may_block(
            (), ("R",), (), blocker.get("colours", ("G",)), blocker.get("types", ("creature",)),
            attacker_protections=(protection,), blocker_subtypes=blocker.get("subtypes", ()),
            blocker_mana_value=blocker.get("mana_value", 2),
        ) is expected

    @pytest.mark.parametrize("keyword, expected", [
        ("protection_from_everything", False), ("protection_from_each_color", False),
        ("protection", True),
    ])
    def test_the_keyword_list_states_everything_and_each_colour(self, keyword, expected):
        assert may_block((keyword,), ("R",), (), ("G",), ("creature",)) is expected


class TestProtectionQuality:
    @pytest.mark.parametrize("fragment, kind, value, keyword", [
        ("Protection from red", QualityKind.COLOUR, "red", "protection_from_red"),
        ("Protection from everything", QualityKind.EVERYTHING, "", "protection_from_everything"),
        ("Protection from each color", QualityKind.CONDITION, "each color",
         "protection_from_each_color"),
        ("Protection:Artifact", QualityKind.CARD_TYPE, "artifact", "protection"),
        ("Protection:Goblin", QualityKind.CREATURE_TYPE, "goblin", "protection"),
        ("Protection:Assembly-Worker", QualityKind.CREATURE_TYPE, "assembly-worker", "protection"),
        ("Protection:Card.MultiColor:multicolored", QualityKind.CONDITION, "multicolored",
         "protection"),
        ("Protection:Card.cmcLE3:mana value 3 or less", QualityKind.CONDITION,
         "mana value 3 or less", "protection"),
    ])
    def test_forge_s_fragment_round_trips(self, fragment, kind, value, keyword):
        quality = ProtectionQuality.parse(fragment)
        assert quality == ProtectionQuality(kind, value)
        assert quality.fragment == fragment and quality.keyword == keyword

    @pytest.mark.parametrize("fragment", [
        "Flying", "Protection:Player.Opponent:each of your opponents",
        "Protection:Card.ChosenColor:chosenColor", "Protection:Creature.Legendary:legendary",
    ])
    def test_a_quality_outside_the_five_kinds_is_none(self, fragment):
        assert ProtectionQuality.parse(fragment) is None

    def test_what_protection_stops(self):
        goblin = Characteristics.made(("creature",), ("goblin",), ("R",), 1)
        elf = Characteristics.made(("creature",), ("elf",), ("G",), 1)
        goblins = (ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin"),)
        assert not may_target(goblins, goblin) and may_target(goblins, elf)
        assert damage_prevented(goblins, goblin) and not damage_prevented(goblins, elf)
        aura = Characteristics.made(("enchantment",), ("aura",), ("B",), 2)
        assert not may_enchant((ProtectionQuality(QualityKind.CARD_TYPE, "enchantment"),), aura)
        assert may_enchant(goblins, aura)
