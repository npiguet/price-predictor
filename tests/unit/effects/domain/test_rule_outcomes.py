"""``rule_outcomes``: each rule function on the cases its former home pinned (T012),
protection's rare qualities (702.16, T137), and the turn-structure rules the
conditions probe reads (500.1, 307.1, 702.61, T161)."""

from __future__ import annotations

import pytest

from effects.domain.rule_outcomes import (
    PHASE_ORDER,
    BlockKeywords,
    Characteristics,
    Combatant,
    EffectKind,
    ProtectionQuality,
    QualityKind,
    damage_lethal,
    damage_prevented,
    forge_phases,
    lethal,
    may_block,
    may_enchant,
    may_target,
    shrink_lethal,
    sorcery_timing,
    split_second_forbids,
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
            attacker, blocker, BlockKeywords(frozenset(keywords[0]), frozenset(keywords[1])),
            step) == expected

    @pytest.mark.parametrize("attacker, blocker, keywords, step, expected", [
        # 704.5f: toughness 0 or less is put into the graveyard, indestructible or not.
        ((1, 1, 0), (0, 0, 0), ({}, {"indestructible"}), "regular", (False, True)),
        ((1, -1, 0), (1, 2, 0), ({"indestructible"}, {}), "first_strike", (True, False)),
        # 702.2b: deathtouch needs damage dealt, and a 0-power source deals none.
        ((0, 3, 0), (3, 3, 0), ({"deathtouch"}, {}), "regular", (True, False)),
        ((0, 4, 0), (3, 3, 0), ({"deathtouch"}, {}), "regular", (False, False)),
        # 704.5g: indestructible stops lethal damage, deathtouch's included.
        ((1, 5, 0), (1, 5, 0), ({"deathtouch"}, {"indestructible"}), "regular", (False, False)),
        # 510.4: two first strikers deal damage together in the first-strike step.
        ((3, 3, 0), (3, 3, 0), ({"first_strike"}, {"first_strike"}), "first_strike", (True, True)),
        ((3, 3, 0), (3, 3, 0), ({"first_strike"}, {"first_strike"}), "regular", (False, False)),
    ])
    def test_the_hard_cases(self, attacker, blocker, keywords, step, expected):
        assert step_deaths(
            attacker, blocker, BlockKeywords(frozenset(keywords[0]), frozenset(keywords[1])),
            step) == expected

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
        assert may_block(Combatant.made(attacker, ("R",)),
                         Combatant.made(blocker, colours, types)) is expected

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
            Combatant.made((), ("R",)),
            Combatant(frozenset(), Characteristics.made(
                blocker.get("types", ("creature",)), blocker.get("subtypes", ()),
                blocker.get("colours", ("G",)), blocker.get("mana_value", 2))),
            attacker_protections=(protection,),
        ) is expected

    @pytest.mark.parametrize("keyword, expected", [
        ("protection_from_everything", False), ("protection_from_each_color", False),
        ("protection", True),
    ])
    def test_the_keyword_list_states_everything_and_each_colour(self, keyword, expected):
        assert may_block(Combatant.made((keyword,), ("R",)),
                         Combatant.made((), ("G",), ("creature",))) is expected


    def test_a_colourless_artifact_attacker_with_intimidate_is_blocked_by_artifacts_only(self):
        """702.13a: only by artifact creatures and creatures sharing a colour with it,
        and a colourless attacker shares none."""
        attacker = Combatant.made(("intimidate",), (), ("artifact", "creature"))
        assert may_block(attacker, Combatant.made((), (), ("artifact", "creature")))
        assert may_block(attacker, Combatant.made((), ("G",), ("artifact", "creature")))
        assert not may_block(attacker, Combatant.made((), ("G",), ("creature",)))
        assert not may_block(attacker, Combatant.made((), (), ("creature",)))

    def test_a_changeling_blocker_is_every_creature_type(self):
        """702.73a: it has the creature type a protection names."""
        goblins = (ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin"),)
        attacker = Combatant.made((), ("R",))
        changeling = Combatant.made(("changeling",), ("G",), ("creature",))
        assert not may_block(attacker, changeling, attacker_protections=goblins)
        assert may_block(attacker, Combatant.made((), ("G",), ("creature",), ("elf",)),
                         attacker_protections=goblins)


class TestProtectionQuality:
    @pytest.mark.parametrize("fragment, kind, value, keyword", [
        ("Protection from red", QualityKind.COLOUR, "red", "protection_from_red"),
        ("Protection from everything", QualityKind.EVERYTHING, "", "protection_from_everything"),
        ("Protection from each color", QualityKind.CONDITION, "each color",
         "protection_from_each_color"),
        ("Protection:Artifact", QualityKind.CARD_TYPE, "artifact", "protection_from_artifact"),
        ("Protection:Goblin", QualityKind.CREATURE_TYPE, "goblin", "protection_from_goblin"),
        ("Protection:Assembly-Worker", QualityKind.CREATURE_TYPE, "assembly-worker",
         "protection_from_assembly-worker"),
        ("Protection:Card.MultiColor:multicolored", QualityKind.CONDITION, "multicolored",
         "protection_from_multicolored"),
        ("Protection:Card.cmcLE3:mana value 3 or less", QualityKind.CONDITION,
         "mana value 3 or less", "protection_from_mana_value_3_or_less"),
    ])
    def test_forge_s_fragment_round_trips(self, fragment, kind, value, keyword):
        quality = ProtectionQuality.parse(fragment)
        assert quality == ProtectionQuality(kind, value)
        assert quality.fragment == fragment and quality.keyword == keyword

    def test_every_whitelisted_quality_round_trips_through_its_keyword(self):
        """A keyword list carries ``quality.keyword``; reading it back gives
        the quality for every one a protection copy or fork can grant."""
        from effects.application.collect_variants import PROTECTION_QUALITY_WHITELIST

        for fragment in PROTECTION_QUALITY_WHITELIST:
            quality = ProtectionQuality.parse(fragment)
            assert ProtectionQuality.of_keyword(quality.keyword) == quality, fragment

    @pytest.mark.parametrize("keyword", [
        "protection_from_creatures", "protection_from_player", "protection",
    ])
    def test_a_keyword_naming_no_quality_reads_as_none(self, keyword):
        assert ProtectionQuality.of_keyword(keyword) is None

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


class TestTurnStructure:
    @pytest.mark.parametrize("value, expected", [
        ("Main1", {"main1"}),
        ("Main", {"main1", "main2"}),
        ("Upkeep->BeginCombat", {"upkeep", "draw", "main1", "combat_begin"}),
        ("BeginCombat->Declare Attackers", {"combat_begin", "combat_declare_attackers"}),
        ("Main2->", {"main2", "end_of_turn", "cleanup"}),
        ("Upkeep,End of Turn", {"upkeep", "end_of_turn"}),
        ("main1", {"main1"}),
    ])
    def test_a_script_phase_list_reads_as_forge_parses_it(self, value, expected):
        assert forge_phases(value) == frozenset(expected)

    @pytest.mark.parametrize("value", [None, "", "Midnight", "Main2->Upkeep"])
    def test_a_phase_list_forge_would_not_read_names_nothing(self, value):
        assert forge_phases(value) is None

    def test_the_order_is_the_turn_s(self):
        assert PHASE_ORDER[0] == "untap" and PHASE_ORDER[-1] == "cleanup"
        assert PHASE_ORDER.index("main1") < PHASE_ORDER.index("combat_begin")

    @pytest.mark.parametrize("own_turn, phase, stack, expected", [
        (True, "main1", 0, True), (True, "main2", 0, True),
        (False, "main1", 0, False), (True, "combat_begin", 0, False),
        (True, "main1", 1, False),
    ])
    def test_sorcery_timing(self, own_turn, phase, stack, expected):
        assert sorcery_timing(own_turn=own_turn, phase=phase, stack_size=stack) is expected

    def test_split_second_forbids_every_candidate_but_a_land_play(self):
        assert split_second_forbids(split_second_on_stack=True, land_play=False)
        assert not split_second_forbids(split_second_on_stack=True, land_play=True)
        assert not split_second_forbids(split_second_on_stack=False, land_play=False)

    def test_split_second_allows_a_special_action(self):
        """702.61b: turning a face-down creature face up, suspending a card and
        the other special actions stay legal (TP4's suspected expected value)."""
        assert not split_second_forbids(split_second_on_stack=True, land_play=False,
                                        special_action=True)
