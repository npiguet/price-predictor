package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.countOfType;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's walk of an ability's sub-ability chain: the descriptions it joins into one line. */
@ExtendWith(ForgeExtension.class)
class RulesParserChainsTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void spellWithSubAbilityDescription() {
        CardFace card = cards.face("a/aetherspouts.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(lineOf(spells.get(0)).contains("each attacking creature"));
    }

    @Test
    void spellWithMultipleSubAbilityDescriptions() {
        // Seed Spark has SpellDescription on both the main SP$ and on a SubAbility SVar.
        // Both sentences are the spell's one line, in chain order.
        CardFace card = cards.face("s/seed_spark.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        Ability root = spells.get(0);
        String text = root.descriptionText();
        assertTrue(text.startsWith("destroy target artifact or enchantment"), text);
        assertTrue(text.contains("create two 1/1 green saproling"), text);
        assertEquals(List.of(), root.subAbilities());
    }

    @Test
    void activatedAbilityWithSubAbilityDescription() {
        // Saprazzan Breaker's activated ability has SpellDescription on both the
        // main AB$ and a sub-ability SVar. Both descriptions are concatenated into
        // the single activated line so one oracle paragraph maps to one output line.
        CardFace card = cards.face("s/saprazzan_breaker.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(), "Should be exactly one activated ability, got: " + card.abilities());
        Ability root = activated.get(0);
        assertTrue(root.descriptionText().contains("mill a card"), "Should contain main description: " + root.descriptionText());
        assertTrue(root.descriptionText().contains("can't be blocked this turn"),
                "Sub-ability text should be concatenated into root description: " + root.descriptionText());
        assertTrue(root.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "No SPELL sub-abilities should remain after concatenation: " + root.subAbilities());
    }

    @Test
    void repeatSubAbilityDescriptionIsIncluded() {
        // March of Souls: main SA destroys all creatures, then a RepeatEach SVar per player
        // creates spirit tokens. The spirit-token text lives in a RepeatSubAbility SVar and
        // must be rendered as part of the single spell's effect tree.
        CardFace card = cards.face("m/march_of_souls.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly one SPELL: " + card.abilities());
        // The rendered text (which calls formatBlock) holds the fully concatenated chain text.
        String formatted = rendered(card);
        assertTrue(formatted.contains("destroy all creatures"),
                "Output should contain 'destroy all creatures': " + formatted);
        assertTrue(formatted.contains("spirit"),
                "Output should contain spirit-token text from RepeatSubAbility: " + formatted);
    }

    @Test
    void activatedAbilityWithDescriptionOnSubAbility() {
        // Arachnus Spinner: activated ability has no SpellDescription on the root SA;
        // the description lives on SubAbility$ DBChange.
        CardFace card = cards.face("a/arachnus_spinner.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(),
                "Arachnus Spinner should have 1 ACTIVATED ability: " + card.abilities());
        String formatted = rendered(card);
        assertTrue(formatted.toLowerCase().contains("search"),
                "Activated ability text should include 'search': " + formatted);
        // The sub-ability chain walk must NOT cause this to be picked up as a
        // spell -- only spells walk the chain for SpellDescription.
        assertEquals(0, countOfType(card, AbilityType.SPELL),
                "Activated ability with sub-ability description should not appear as spell");
    }

    @ParameterizedTest
    @CsvSource({"f/fleshformer.txt", "g/grinning_totem.txt"})
    void activatedAbilitySubChainConcatenatedNotSplit(String file) {
        // Oracle has one line, so the converter must emit 1 activated line, not 2
        // (root + sub-ability): the sub-ability text is concatenated into the root line.
        CardFace card = cards.face(file);
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(), file + " should have exactly 1 ACTIVATED: " + card.abilities());
        long spellSubs = activated.get(0).subAbilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).count();
        assertEquals(0, spellSubs,
                "No SPELL sub-abilities should remain after concatenation: " + activated.get(0).subAbilities());
    }

    @Test
    void sheoldredRestorationEmitsSpellEffect() {
        // TriggerDescription$ on root spell SA must be used as the spell description
        var spells = abilitiesOfType(cards.face("s/sheoldreds_restoration.txt"), AbilityType.SPELL);
        assertTrue(spells.stream().anyMatch(
                a -> a.descriptionText().toLowerCase().contains("return target creature card")),
                "sheoldred's restoration spell effect missing: " + spells);
    }

    @Test
    void triggerExecuteChainTrailingTextConcatenated() {
        // Oil-Gorger Troll: execute chain (TrigGainLife → DBDraw) has SpellDescription
        // that must be concatenated into the trigger description.
        CardFace card = cards.face("o/oil_gorger_troll.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have TRIGGERED: " + card.abilities());
        Ability t = triggered.get(0);
        assertTrue(t.descriptionText().contains("you gain 3 life"),
                "Must contain first action: " + t.descriptionText());
        assertTrue(t.descriptionText().contains("oil counter"),
                "Must contain trailing execute chain text: " + t.descriptionText());
    }

    @Test
    void effectTriggerDescriptionEmittedAsSecondSpellLine() {
        // Mage Hunters' Onslaught: DB$ Effect with Triggers$ TrigBlocking.
        // TriggerDescription of TrigBlocking must appear as a second spell line.
        CardFace card = cards.face("m/mage_hunters_onslaught.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty(), "Should have SPELL: " + card.abilities());
        String formatted = rendered(card);
        assertTrue(formatted.contains("destroy target creature or planeswalker"),
                "Must contain first effect: " + formatted);
        assertTrue(formatted.contains("whenever a creature blocks"),
                "Must contain trigger description from Effect SVar: " + formatted);
    }

    @Test
    void effectTriggerDescriptionConcatenatedIntoSpellLine() {
        // Reckless Blaze: DB$ Effect with Triggers$ DiesTrig.
        // TriggerDescription must appear in the output (oracle=1 single paragraph).
        CardFace card = cards.face("r/reckless_blaze.txt");
        String formatted = rendered(card);
        assertTrue(formatted.contains("deals 5 damage to each creature"),
                "Must contain main effect: " + formatted);
        assertTrue(formatted.contains("dies this turn"),
                "Must contain trigger description from Effect SVar: " + formatted);
    }

    @Test
    void effectReplacementDescriptionConcatenated() {
        // Dazzling Reflection: DB$ Effect with ReplacementEffects$ Dazzle.
        // Dazzle's Description$ must appear in the output.
        CardFace card = cards.face("d/dazzling_reflection.txt");
        String formatted = rendered(card);
        assertTrue(formatted.contains("you gain life equal to target creature"),
                "Must contain main effect: " + formatted);
        assertTrue(formatted.contains("prevent that damage"),
                "Must contain replacement Description$ from Effect SVar: " + formatted);
    }

    @Test
    void stackDescriptionFallbackForSubAbility() {
        // Transgress the Mind: ExileCard SVar has StackDescription but no SpellDescription.
        // Must concatenate the StackDescription text into the spell line.
        CardFace card = cards.face("t/transgress_the_mind.txt");
        String formatted = rendered(card);
        assertTrue(formatted.contains("target player reveals their hand"),
                "Must contain root spell description: " + formatted);
        assertTrue(formatted.contains("mana value 3 or greater"),
                "Must contain sub-ability StackDescription text: " + formatted);
    }
}
