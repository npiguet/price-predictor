package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static com.pricepredictor.connector.ConvertedTextAssertions.assertOccursOnce;
import static com.pricepredictor.connector.ConvertedTextAssertions.occurrences;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's guards against printing one oracle clause twice. */
@ExtendWith(ForgeExtension.class)
class RulesParserDuplicatesTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void etbReplacementNoDuplicateSpellLine() {
        // Sigarda's Splendor: has an ETBReplacement/ImmediateTrigger SVar that the
        // triggers/replacements loop processes and the spell loop must not process
        // again, so no ability text should appear twice.
        CardFace card = cards.face("s/sigardas_splendor.txt");
        String formatted = rendered(card);
        // Collect all non-header values from formatted output
        long duplicateCount = java.util.Arrays.stream(formatted.split("\n"))
                .filter(line -> line.contains(":"))
                .map(line -> {
                    int colon = line.indexOf(':');
                    return line.substring(colon + 1).strip();
                })
                .collect(java.util.stream.Collectors.groupingBy(v -> v, java.util.stream.Collectors.counting()))
                .values().stream().filter(count -> count > 1).count();
        assertEquals(0, duplicateCount,
                "sigarda's_splendor should have no duplicate ability lines: " + formatted);
    }

    // Regression: delayed-trigger execute SVars must not bleed into the trigger description.
    // DelTrigBlocked/DelTrigBlocker have TriggerDescription$ which is already summarised
    // in the root TriggerDescription; concatenating it would duplicate the text.
    @Test
    void delayedTriggerExecuteNotConcatenatedIntoTrigger() {
        CardFace card = cards.face("s/sawtooth_ogre.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have TRIGGERED: " + card.abilities());
        for (Ability t : triggered) {
            assertOccursOnce(t.descriptionText(), "1 damage");
        }
    }

    @Test
    void delayedTriggerDestroyNotConcatenatedIntoTrigger() {
        CardFace card = cards.face("t/tangle_asp.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have TRIGGERED: " + card.abilities());
        for (Ability t : triggered) {
            assertOccursOnce(t.descriptionText(), "destroy");
        }
    }

    // Regression: StackDescription on a sub-ability that copies parent SpellDescription verbatim
    // must not produce a duplicate spell line (e.g. Bifurcate DBChangeZone StackDescription).
    @Test
    void stackDescriptionDuplicatingParentNotEmitted() {
        CardFace card = cards.face("b/bifurcate.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        // A duplicated StackDescription would print the clause twice.
        assertOccursOnce(spells.get(0).descriptionText(), "search");
    }

    // Regression: replacement-effect Description$ that exactly matches parent SpellDescription
    // must not be emitted as a sub-ability (e.g. Shadowbane RepDmg.Description$).
    @Test
    void replacementDescriptionDuplicatingParentNotEmitted() {
        CardFace card = cards.face("s/shadowbane.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        // No sub-abilities — the replacement text is identical to the spell description
        assertEquals(0, spells.get(0).subAbilities().size(),
                "Duplicate replacement Description$ must not produce sub-ability: " + spells.get(0));
    }

    // Regression: replacement-effect Description$ that is a suffix of the SpellDescription
    // must not be emitted again (e.g. Energy Arc RPrevent1).
    @Test
    void replacementDescriptionContainedInParentNotEmitted() {
        CardFace card = cards.face("e/energy_arc.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        assertOccursOnce(spells.get(0).descriptionText(), "prevent");
    }

    // Regression: replacement-effect Description$ containing Forge placeholder EFFECTSOURCE
    // must not be emitted as oracle text (e.g. Delirium RPrevent1).
    @Test
    void replacementDescriptionWithEffectsourceNotEmitted() {
        CardFace card = cards.face("d/delirium.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        // Once, from the SpellDescription, and never again from the replacement SVar.
        assertOccursOnce(rendered(card), "prevent");
    }

    // Regression: a trigger SVar referenced in DB$ Effect.Triggers$ that has TriggerZones$
    // must not be emitted as a spell sub-ability (it's also a top-level T: trigger).
    @Test
    void triggerZonesSVarNotEmittedAsSpellSubAbility() {
        CardFace card = cards.face("e/ertais_meddling.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty(), "Should have SPELL: " + card.abilities());
        // The spell sub-ability should not duplicate the triggered ability
        for (Ability spell : spells) {
            assertFalse(spell.descriptionText().contains("beginning of each"),
                    "TriggerZones trigger must not be embedded in spell line: " + spell.descriptionText());
            assertTrue(spell.subAbilities().isEmpty(),
                    "Spell must not have sub-abilities from zone-scoped triggers: " + spell);
        }
    }

    // Regression: identical SpellDescription values across a sub-ability chain must all
    // be emitted — each represents a distinct oracle effect (e.g. Bounty of Might x3).
    @Test
    void identicalSpellDescriptionsInChainAllEmitted() {
        CardFace card = cards.face("b/bounty_of_might.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        // The single SPELL ability has a nested sub-ability chain representing all 3 effects.
        // Rendering must produce all 3 repetitions (not deduplicated away).
        String formatted = rendered(card);
        assertEquals(3, occurrences(formatted, "+3/+3"),
                "All 3 identical pump effects must be present: " + formatted);
    }

    // Regression: replacement-effect Description$ containing EFFECTSOURCE in an activated ability
    // must not be emitted even when ActivatedAbilityEntry calls fromChain() without parentDesc.
    @Test
    void replacementDescriptionWithEffectsourceNotEmittedFromActivatedAbility() {
        // Stuffy Doll Avatar: SelflessDamage.Description$ contains EFFECTSOURCE placeholder.
        // ActivatedAbilityEntry calls fromChain(sub) without parentDesc so parentDesc=null;
        // the effectsource guard must fire before the parentDesc null check.
        CardFace card = cards.face("s/stuffy_doll_avatar.txt");
        String formatted = rendered(card);
        assertFalse(formatted.contains("effectsource"),
                "EFFECTSOURCE placeholder must not appear in output: " + formatted);
    }

    // Regression: DB$ Effect SA whose SpellDescription is entirely reminder text (strips to empty)
    // must not trigger collectEffectDescriptions — its replacement children are already covered.
    @Test
    void replacementDescriptionOfAllReminderTextEffectNotEmitted() {
        // Sarah's Wings: NoDamage SVar has SpellDescription$ = "(Players with flying can't...)"
        // which is entirely reminder text. collectEffectDescriptions must not run on it.
        CardFace card = cards.face("s/sarahs_wings.txt");
        String formatted = rendered(card);
        assertTrue(formatted.contains("flying"),
                "Must contain flying grant: " + formatted);
        assertFalse(formatted.contains("prevent all damage"),
                "RPrevent.Description must not be emitted as extra ability: " + formatted);
    }

    // Regression: replacement-effect Description$ using "this card" instead of CARDNAME
    // must be treated as redundant with the parent SpellDescription that uses CARDNAME.
    @Test
    void replacementDescriptionWithThisCardEquivalentToCardname() {
        // Eye for an Eye: SelflessDamage.Description$ says "this card deals that much damage"
        // while SpellDescription says "CARDNAME deals that much damage" — same meaning.
        CardFace card = cards.face("e/eye_for_an_eye.txt");
        String formatted = rendered(card);
        long count = formatted.lines()
                .filter(l -> l.contains("deals that much damage to that source"))
                .count();
        assertEquals(1, count,
                "The deals-damage effect must appear exactly once (not doubled): " + formatted);
    }
}
