package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.aggregator.AggregateWith;
import org.junit.jupiter.params.aggregator.ArgumentsAccessor;
import org.junit.jupiter.params.aggregator.ArgumentsAggregator;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertContainsAll;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertHasAbility;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.countOfType;
import static com.pricepredictor.connector.FaceAbilities.firstIndexOf;
import static com.pricepredictor.connector.FaceAbilities.lastIndexOf;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's cost lines: additional, alternate and reduced costs, their order and their text. */
@ExtendWith(ForgeExtension.class)
class RulesParserCostsTest {

    private static final Set<AbilityType> COST_TYPES = EnumSet.of(
            AbilityType.ADDITIONAL_COST, AbilityType.ALTERNATE_COST, AbilityType.COST_REDUCTION);

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void mayFlashCostUsesReminderText() {
        CardFace card = cards.face("a/asinine_antics.txt");
        var costs = abilitiesOfType(card, AbilityType.ADDITIONAL_COST);
        assertEquals(1, costs.size());
        assertTrue(costs.get(0).descriptionText().contains("you may cast CARDNAME as though it had flash"),
                "MayFlashCost should use reminder text: " + costs.get(0).descriptionText());
    }

    @Test
    void spellWithAdditionalCost() {
        CardFace card = cards.face("a/abandon_hope.txt");

        // The cost is the first sentence of the spell's one line.
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), card.abilities().toString());
        String line = lineOf(spells.get(0));
        assertTrue(line.startsWith("spell[1]: as an additional cost to cast this spell, discard X"), line);
        assertTrue(line.contains(". look at target opponent's hand"), line);
    }

    @Test
    void cleaveEmittedAsAlternateCost() {
        CardFace card = cards.face("a/alchemists_gambit.txt");

        String altLine = lineOf(abilitiesOfType(card, AbilityType.ALTERNATE_COST).get(0));
        assertTrue(altLine.contains("cleave"));
        assertTrue(altLine.contains("{4}{U}{U}{R}"));
        assertFalse(altLine.matches(".*\\{R}\\d.*"), "No trailing digits after cost");

        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty());
        // Verify sequential action numbering in formatted output
        String output = rendered(card);
        for (int i = 0; i < spells.size(); i++) {
            assertTrue(output.contains("spell[" + (i + 1) + "]:"),
                    "Expected spell[" + (i + 1) + "] in output: " + output);
        }
    }

    @Test
    void creatureWithAdditionalCost() {
        CardFace card = cards.face("a/abhorrent_oculus.txt");
        var costs = abilitiesOfType(card, AbilityType.ADDITIONAL_COST);
        assertEquals(1, costs.size());
        assertTrue(lineOf(costs.get(0)).contains("exile"));
    }

    @Test
    void additionalCostFromRaiseCostStatic() {
        CardFace card = cards.face("a/aether_tide.txt");

        assertCostLineWithoutItsLeadIn(card, "discard X creature cards");
        assertEquals(0, countOfType(card, AbilityType.STATIC), "RaiseCost should not appear as static");

        assertSpellLineAfterTheCosts(card, "return X target creatures");
    }

    @Test
    void raiseCostOnOtherSpellsRemainsStatic() {
        CardFace card = cards.face("a/aura_of_silence.txt");
        assertEquals(1, abilitiesOfType(card, AbilityType.STATIC).size());
        assertTrue(abilitiesOfType(card, AbilityType.STATIC).get(0).descriptionText().contains("cost {2} more to cast"));
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void optionalAdditionalCost() {
        CardFace card = cards.face("a/analyze_the_pollen.txt");

        assertCostLineWithoutItsLeadIn(card, "you may collect evidence 8");
        assertEquals(0, countOfType(card, AbilityType.STATIC));
        assertCostsBeforeSpells(card);
    }

    @Test
    void alternateAdditionalCost() {
        CardFace card = cards.face("a/annihilating_glare.txt");

        assertContainsAll(additionalCostLine(card), "sacrifice", "or");
        assertEquals(0, countOfType(card, AbilityType.STATIC)
                + countOfType(card, AbilityType.ACTIVATED),
                "AlternateAdditionalCost should not appear as keyword");

        assertSpellLineAfterTheCosts(card, "destroy target creature or planeswalker");
    }

    @Test
    void doubleKickerIncludesBothCosts() {
        assertContainsAll(additionalCostLine(cards.face("a/archangel_of_wrath.txt")),
                "kicker", "{B}", "{R}", "and/or");
    }

    @ParameterizedTest
    @CsvSource({
            "y/yuffie_materia_hunter.txt, ninjutsu",
            "v/vizier_of_many_faces.txt,  embalm",
            "z/zephyrim.txt,              miracle",
            "y/yotian_frontliner.txt,     unearth",
            "s/sheer_drop.txt,            awaken",
            "w/wheel_of_fate.txt,         suspend",
            "s/surge_of_acclaim.txt,      jump-start",
    })
    void keywordClassifiedAsAlternateCost(@AggregateWith(ScriptText.FromCsv.class) ScriptText keyword) {
        assertHasAbility(cards.face(keyword.file()), AbilityType.ALTERNATE_COST, keyword.text());
    }

    @ParameterizedTest
    @CsvSource({
            "t/troublemaker_ouphe.txt,  bargain",
            "t/traitors_roar.txt,       conspire",
            "w/wear_away.txt,           splice",
            "u/unfortunate_accident.txt, spree",
    })
    void keywordClassifiedAsAdditionalCost(@AggregateWith(ScriptText.FromCsv.class) ScriptText keyword) {
        assertHasAbility(cards.face(keyword.file()), AbilityType.ADDITIONAL_COST, keyword.text());
    }

    @ParameterizedTest
    @CsvSource({
            "s/sojourners_companion.txt, affinity",
            "w/will_of_the_naga.txt,    delve",
            "w/whir_of_invention.txt,   improvise",
    })
    void keywordClassifiedAsCostReduction(@AggregateWith(ScriptText.FromCsv.class) ScriptText keyword) {
        assertHasAbility(cards.face(keyword.file()), AbilityType.COST_REDUCTION, keyword.text());
    }

    @Test
    void convokeClassifiedAsCostReduction() {
        CardFace card = cards.convert(
                "Name:Test Convoke Spell", "ManaCost:3 W W", "Types:Sorcery",
                "K:Convoke", "A:SP$ Destroy | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "Oracle:").faces().get(0);
        assertHasAbility(card, AbilityType.COST_REDUCTION, "convoke");
    }

    @Test
    void allCostTypesSortBeforeOtherAbilities() {
        // Whir of Invention: cost reduction should be first ability
        CardFace whir = cards.face("w/whir_of_invention.txt");
        assertTrue(whir.abilities().size() >= 2);
        assertEquals(AbilityType.COST_REDUCTION, whir.abilities().get(0).type());

        // Zephyrim: alternate cost + additional cost + keywords
        CardFace zeph = cards.face("z/zephyrim.txt");
        int firstNonCostIdx = firstIndexOf(zeph, t -> !COST_TYPES.contains(t));
        int lastCostIdx = lastIndexOf(zeph, COST_TYPES::contains);
        assertTrue(lastCostIdx >= 0, "Zephyrim has cost lines: " + zeph.abilities());
        assertTrue(firstNonCostIdx >= 0, "Zephyrim has non-cost lines: " + zeph.abilities());
        assertTrue(lastCostIdx < firstNonCostIdx,
                "All cost lines should appear before non-cost lines: " + zeph.abilities());
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            // K:Equip:3:::ReduceCost$ Y — a fixed reduction counted by other equipment
            "Plate Armor equip,              p/plate_armor.txt,                    equip,         costs {1} less, each other equipment",
            // ReduceCost$ X — a variable reduction
            "Belt of Giant Strength equip,   b/belt_of_giant_strength.txt,         equip,         costs {X} less, power of the creature",
            // K:Specialize:5::This ability costs {3} less…:ReduceCost$ X
            "Imoen specialize,               i/imoen_trickster_friend.txt,         specialize,    costs {3} less, instant and/or sorcery cards in your graveyard",
    })
    void keywordReduceCostAppendsText(@AggregateWith(ReducedCostLine.FromCsv.class) ReducedCostLine line) {
        assertActivatedLineCarries(line);
    }

    /**
     * Adapt and Monstrosity are not keywords here but scripted abilities
     * ({@code A:AB$ PutCounter | Adapt$ 4 | ReduceCost$ X | SpellDescription$ …}),
     * so their reduction reaches the line through the ability's own
     * description rather than through the keyword's appended text.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            // A:AB$ PutCounter | Cost$ 7 U | Adapt$ 4 | ReduceCost$ X
            "Pteramander adapt,              p/pteramander.txt,                    adapt,         costs {1} less, instant and sorcery card in your graveyard",
            // A:AB$ PutCounter | Cost$ 6 B | Adapt$ 4 | ReduceCost$ X
            "Etherium Pteramander adapt,     e/etherium_pteramander.txt,           adapt,         costs {1} less, other artifact you control",
            // A:AB$ PutCounter | Cost$ 10 B G | Monstrosity$ 10 | ReduceCost$ X
            "Grim Giganotosaurus monstrosity, g/grim_giganotosaurus.txt,           monstrosity,   costs {1} less, creature with power 4",
    })
    void scriptedReduceCostKeepsItsSpellDescription(@AggregateWith(ReducedCostLine.FromCsv.class) ReducedCostLine line) {
        assertActivatedLineCarries(line);
    }

    /**
     * One reduced-cost case: the activated line of the card in {@code file}
     * naming {@code keyword} carries {@code reduction} and {@code countClause}.
     */
    record ReducedCostLine(String file, String keyword, String reduction, String countClause) {

        /** Reads a case from a CSV row whose first column only names the case. */
        static final class FromCsv implements ArgumentsAggregator {
            @Override
            public ReducedCostLine aggregateArguments(ArgumentsAccessor row, ParameterContext context) {
                return new ReducedCostLine(row.getString(1), row.getString(2), row.getString(3), row.getString(4));
            }
        }
    }

    private void assertActivatedLineCarries(ReducedCostLine line) {
        Ability ability = abilitiesOfType(cards.face(line.file()), AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains(line.keyword()))
                .findFirst().orElseThrow();
        assertTrue(ability.descriptionText().contains(line.reduction()),
                "Must include cost reduction: " + ability.descriptionText());
        assertTrue(ability.descriptionText().contains(line.countClause()),
                "Must include count clause: " + ability.descriptionText());
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            // K:Craft:3 U ExileCtrlOrGrave<1/Island.Other>
            "Waterlogged Hulk island,           w/waterlogged_hulk_watertight_gondola.txt,        craft with island,                    {3}{U}",
            // K:Craft:3 R R ExileCtrlOrGrave<1/Artifact.Other>
            "Dire Flail multicolour cost,       d/dire_flail_dire_blunderbuss.txt,                craft with artifact,                  {3}{R}{R}",
            // K:Craft:5 XMin1 ExileCtrlOrGrave<X/Permanent.Other/permanent>
            "Sunbird Standard variable count,   s/sunbird_standard_sunbird_effigy.txt,            craft with one or more,               {5}",
            // K:Craft:4 ExileCtrlOrGrave<…>:a Dinosaur, a Merfolk, a Pirate, and a Vampire:the four
            "Throne explicit multi-type text,   t/throne_of_the_grim_captain_the_grim_captain.txt, craft with a dinosaur,               {4}",
    })
    void craftEmitsTheTerseFormat(@AggregateWith(CraftLine.FromCsv.class) CraftLine line) {
        Ability craft = abilitiesOfType(cards.face(line.file()), AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains(line.craftText()),
                "Must include '" + line.craftText() + "': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains(line.manaCost()),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    /** One craft case: the craft line of the card in {@code file} reads {@code craftText} and costs {@code manaCost}. */
    record CraftLine(String file, String craftText, String manaCost) {

        /** Reads a case from a CSV row whose first column only names the case. */
        static final class FromCsv implements ArgumentsAggregator {
            @Override
            public CraftLine aggregateArguments(ArgumentsAccessor row, ParameterContext context) {
                return new CraftLine(row.getString(1), row.getString(2), row.getString(3));
            }
        }
    }

    @Test
    void crashingWaveAdditionalCostPresentSpellEffectFromTheOracle() {
        // A:SP$ Tap has no SpellDescription: its effect is the oracle sentence the static's
        // waterbend line does not state.
        var card = cards.face("c/crashing_wave.txt");
        assertEquals(1, countOfType(card, AbilityType.ADDITIONAL_COST));
        assertTrue(abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0)
                   .descriptionText().contains("waterbend"));
        assertEquals(List.of("tap up to X target creatures, then distribute three stun counters"
                        + " among tapped creatures your opponents control."),
                abilitiesOfType(card, AbilityType.SPELL).stream().map(Ability::descriptionText).toList());
    }

    /** The line of the face's first additional cost. */
    private static String additionalCostLine(CardFace card) {
        return lineOf(abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0));
    }

    /** The first additional cost line holds {@code costText} without the "as an additional cost" lead-in. */
    private static void assertCostLineWithoutItsLeadIn(CardFace card, String costText) {
        String costLine = additionalCostLine(card);
        assertTrue(costLine.contains(costText), costLine);
        assertFalse(costLine.contains("as an additional cost to cast"), costLine);
    }

    /** The first spell line holds {@code spellText}, and every cost line comes before the spell lines. */
    private static void assertSpellLineAfterTheCosts(CardFace card, String spellText) {
        assertTrue(lineOf(abilitiesOfType(card, AbilityType.SPELL).get(0)).contains(spellText));
        assertCostsBeforeSpells(card);
    }

    private static void assertCostsBeforeSpells(CardFace card) {
        var abilities = card.abilities();
        int lastCostIdx = lastIndexOf(card, COST_TYPES::contains);
        int firstSpellIdx = firstIndexOf(card, t -> t == AbilityType.SPELL);
        // A card missing either has no order to check, and would pass silently.
        assertTrue(lastCostIdx >= 0, "no cost line: " + abilities);
        assertTrue(firstSpellIdx >= 0, "no spell line: " + abilities);
        assertTrue(lastCostIdx < firstSpellIdx,
                "Costs should appear before spells: " + abilities);
    }
}
