package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.allOptions;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's dice rolls: the roll's line and its outcomes as option lines. */
@ExtendWith(ForgeExtension.class)
class RulesParserDiceRollsTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void diceRollEmitsOutcomesWithVertReplaced() {
        // Cone of Cold rolls a d20 with 3 outcome ranges: one SPELL line
        // holding the roll, its 3 outcomes OPTION lines below it, with | for VERT.
        CardFace card = cards.face("c/cone_of_cold.txt");
        var lines = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, lines.size(), "Dice roll card should emit 1 SPELL: " + card.abilities());
        assertTrue(lines.get(0).descriptionText().contains("roll a d20"),
                "Should have 'roll a d20' spell: " + lines);
        var spells = allOptions(card);
        assertEquals(3, spells.size(), "3 outcome options: " + card.abilities());
        assertTrue(spells.stream().anyMatch(a -> a.descriptionText().contains("1") && a.descriptionText().contains("tap all")),
                "Should have 1-9 outcome: " + spells);
        assertTrue(spells.stream().anyMatch(a -> a.descriptionText().contains("10") && a.descriptionText().contains("don't untap")),
                "Should have 10-19 outcome: " + spells);
        assertTrue(spells.stream().anyMatch(a -> a.descriptionText().contains("20") && a.descriptionText().contains("don't untap")),
                "Should have 20 outcome: " + spells);
        assertTrue(spells.stream().noneMatch(a -> a.descriptionText().contains("VERT")),
                "No outcome description should contain 'VERT': " + spells);
        assertTrue(spells.stream().anyMatch(a -> a.descriptionText().contains("|")),
                "Outcome descriptions should use '|' separator: " + spells);
    }

    @Test
    void aSpellsRollUnderARepeatPrintsItsOutcomes() {
        // Strategy, Schmategy: SP → Repeat → RepeatSubAbility$ RollDice with
        // six outcomes. The roll is reached through the repeat, not the
        // sub-ability chain, and its outcomes still print, as options.
        CardFace card = cards.face("s/strategy_schmategy.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "The roll: " + spells);
        assertTrue(spells.get(0).descriptionText().contains("roll a six-sided die"), spells.toString());
        var outcomes = allOptions(card);
        assertEquals(6, outcomes.size(), "Six outcomes: " + outcomes);
        assertTrue(outcomes.get(0).descriptionText().contains("do nothing"), outcomes.toString());
        assertTrue(outcomes.get(5).descriptionText().contains("repeat this process two more times"),
                outcomes.toString());
    }

    @Test
    void anActivatedRollUnderARepeatPrintsItsOutcomesAsOptions() {
        // No printed card activates a roll under a repeat, so the script is
        // synthetic; the walk is the one spells and triggers use.
        Ability act = abilitiesOfType(cards.convert(
                "Name:Repeat Roller", "ManaCost:2", "Types:Artifact",
                "A:AB$ Repeat | Cost$ T | RepeatSubAbility$ DBRoll | RepeatCount$ 2"
                        + " | SpellDescription$ Roll a d6 twice.",
                "SVar:DBRoll:DB$ RollDice | Sides$ 6 | ResultSubAbilities$ 1-3:DBDraw,4-6:DBGain",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | SpellDescription$ 1-3 VERT Draw a card.",
                "SVar:DBGain:DB$ GainLife | LifeAmount$ 2 | SpellDescription$ 4-6 VERT Gain 2 life.",
                "Oracle:").faces().get(0), AbilityType.ACTIVATED).get(0);
        List<Ability> options = act.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(List.of("1-3 | draw a card.", "4-6 | gain 2 life."),
                options.stream().map(Ability::descriptionText).toList());
    }

    @Test
    void activatedDiceRollEmitsOutcomesAsOptions() {
        // Treasure Chest: activated RollDice with 4 result ranges.
        // Expected: 1 ACTIVATED with a single-line description and 4 OPTION sub-abilities.
        CardFace card = cards.face("t/treasure_chest.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(), "Should have exactly 1 ACTIVATED: " + card.abilities());
        Ability act = activated.get(0);
        assertFalse(act.descriptionText().contains("\n"),
                "Activated description must be single-line (no embedded newlines): " + act.descriptionText());
        assertTrue(act.descriptionText().contains("roll a d20"),
                "Activated description should contain 'roll a d20': " + act.descriptionText());
        List<Ability> options = act.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(4, options.size(), "Should have 4 OPTION sub-abilities: " + act.subAbilities());
        assertTrue(options.stream().anyMatch(a -> a.descriptionText().contains("1") && a.descriptionText().contains("lose 3 life")),
                "Should have '1 | ... lose 3 life' option: " + options);
        assertTrue(options.stream().anyMatch(a -> a.descriptionText().contains("2") && a.descriptionText().contains("treasure")),
                "Should have '2—9 | ... treasure' option: " + options);
        assertTrue(options.stream().anyMatch(a -> a.descriptionText().contains("10") && a.descriptionText().contains("gain 3 life")),
                "Should have '10—19 | ... gain 3 life' option: " + options);
        assertTrue(options.stream().anyMatch(a -> a.descriptionText().contains("20") && a.descriptionText().contains("library")),
                "Should have '20 | ... library' option: " + options);
    }

    @Test
    void triggeredDiceRollEmitsOutcomesAsOptions() {
        // Swarming Goblins: triggered RollDice with 3 result ranges.
        // Expected: 1 TRIGGERED with roll-a-d20 header and 3 OPTION sub-abilities.
        CardFace card = cards.face("s/swarming_goblins.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have at least 1 TRIGGERED: " + card.abilities());
        // Find the one with dice-roll options
        Ability diceTriggered = triggered.stream()
                .filter(a -> !a.subAbilities().stream()
                        .filter(s -> s.type() == AbilityType.OPTION).toList().isEmpty())
                .findFirst()
                .orElse(null);
        assertNotNull(diceTriggered,
                "Should have a TRIGGERED with OPTION sub-abilities: " + triggered);
        List<Ability> options = diceTriggered.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(3, options.size(),
                "Should have one OPTION sub-ability per dice result range: " + options);
    }

    @Test
    void diceRollTriggerExpandsAbilityPlaceholder() {
        // Delina: Trigger → Repeat → RollDice, TriggerDescription ends with "ABILITY"
        CardFace card = cards.face("d/delina_wild_mage.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.get(0);
        assertFalse(t.descriptionText().contains("ABILITY"), "ABILITY must be replaced: " + t.descriptionText());
        assertTrue(t.descriptionText().contains("roll"), "Must mention dice roll: " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(2, options.size(), "Must have 2 dice-roll options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION));
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("14")));
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("20")));
    }
}
