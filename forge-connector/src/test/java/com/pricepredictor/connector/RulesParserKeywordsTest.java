package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.aggregator.AggregateWith;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertHasAbility;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertNoRawEtbReplacementKeyword;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.countOfType;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's keyword lines: how each keyword is classified and what it prints. */
@ExtendWith(ForgeExtension.class)
class RulesParserKeywordsTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void passiveKeywords() {
        var keywords = abilitiesOfType(cards.face("s/serra_angel.txt"), AbilityType.STATIC);
        assertTrue(keywords.stream().anyMatch(k -> k.descriptionText().equals("flying")));
        assertTrue(keywords.stream().anyMatch(k -> k.descriptionText().equals("vigilance")));
        assertFalse(lineOf(keywords.get(0)).contains("["), "Static keywords should have no action number");
    }

    @Test
    void internalKeywordUsesReminderText() {
        // MayFlashSac is a Forge-internal keyword with a camelCase name.
        // The converter should use its reminder text as the description.
        CardFace card = cards.face("l/lightning_reflexes.txt");
        assertTrue(card.abilities().stream().anyMatch(
                a -> a.descriptionText().contains("you may cast CARDNAME as though it had flash")),
                "MayFlashSac should use reminder text: " + card.abilities());
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            // K:Protection:Creature must read "protection from creature", not Forge's
            // internal "protection:creature" filter syntax.
            "a colour; a/animar_soul_of_elements.txt; protection from white|protection from black",
            "a card type; b/beloved_chaplain.txt; protection from",
            // K:Protection:Card.MultiColor:multicolored must not leak the filter expression.
            "a filter expression; e/enemy_of_the_guildpact.txt; protection from multicolored",
    })
    void protectionKeywordReadsAsProtectionFromTheTarget(
            @AggregateWith(ScriptText.FromNamedCsv.class) ScriptText protection) {
        var statics = abilitiesOfType(cards.face(protection.file()), AbilityType.STATIC);
        for (String expected : protection.text().split("\\|")) {
            assertTrue(statics.stream().anyMatch(k -> k.descriptionText().contains(expected)),
                    "expected '" + expected + "' in: " + statics);
        }
    }

    @Test
    void etbReplacementOnNonClassCard() {
        CardFace card = cards.face("f/flickering_ward.txt");
        assertNoRawEtbReplacementKeyword(card);
        assertTrue(abilitiesOfType(card, AbilityType.REPLACEMENT).stream()
                .anyMatch(r -> r.descriptionText().contains("choose a color")));
    }

    @Test
    void etbCounterWithDescriptionFallback() {
        CardFace card = cards.face("a/ambitious_dragonborn.txt");
        assertNoRawEtbReplacementKeyword(card);
        var replacements = abilitiesOfType(card, AbilityType.REPLACEMENT);
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).descriptionText().contains("CARDNAME enters with X +1/+1 counters"));
    }

    @Test
    void companionKeywordIncludesRestriction() {
        assertHasAbility(cards.face("g/gyruda_doom_of_depths.txt"),
                AbilityType.STATIC, "companion", "even mana value");
    }

    @Test
    void giftKeywordIncludesParameter() {
        assertHasAbility(cards.face("d/dawns_truce.txt"), AbilityType.STATIC, "gift a card");
    }

    @Test
    void draftOnlyCardNoDuplicateTextFallback() {
        // Cogwork Librarian has only Draft: fields and no game-engine abilities.
        // It should emit exactly those 2 draft lines — the oracle fallback must NOT
        // also fire and add the same content again as text: lines.
        CardFace card = cards.face("c/cogwork_librarian.txt");
        var draft = abilitiesOfType(card, AbilityType.DRAFT);
        assertEquals(2, draft.size(), "Should have exactly 2 draft lines: " + card.abilities());
        assertEquals(0, countOfType(card, AbilityType.TEXT),
                "Should have no text: fallback duplicates: " + card.abilities());
        assertEquals(2, card.abilities().size(),
                "Total abilities should be exactly 2: " + card.abilities());
    }

    @Test
    void draftLinesEmittedBeforeOtherAbilities() {
        // Aether Searcher has two Draft: lines and one triggered ability.
        // All three should appear; draft lines should precede the triggered ability.
        CardFace card = cards.face("a/aether_searcher.txt");
        var draft = abilitiesOfType(card, AbilityType.DRAFT);
        assertEquals(2, draft.size(), "Should have 2 draft lines: " + card.abilities());
        assertTrue(draft.get(0).descriptionText().contains("reveal CARDNAME as you draft it"),
                "First draft line: " + draft.get(0).descriptionText());
        assertTrue(draft.get(1).descriptionText().contains("reveal the next card you draft"),
                "Second draft line: " + draft.get(1).descriptionText());
        // Triggered ability should also still be present
        assertFalse(abilitiesOfType(card, AbilityType.TRIGGERED).isEmpty(),
                "Triggered ability should also be emitted: " + card.abilities());
        // Draft lines must come first
        int lastDraftIdx = -1, firstTriggeredIdx = -1;
        for (int i = 0; i < card.abilities().size(); i++) {
            AbilityType t = card.abilities().get(i).type();
            if (t == AbilityType.DRAFT) lastDraftIdx = i;
            else if (t == AbilityType.TRIGGERED && firstTriggeredIdx == -1) firstTriggeredIdx = i;
        }
        assertTrue(lastDraftIdx < firstTriggeredIdx,
                "Draft lines must precede triggered ability: " + card.abilities());
    }

    @Test
    void hauntNonCreatureEmitsSpellEffectAndTriggers() {
        // Seize the Soul is a non-creature haunt spell.
        // Expected: 1 SPELL (the primary effect) + 2 TRIGGERED (haunt keyword + haunted-dies trigger).
        CardFace card = cards.face("s/seize_the_soul.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, spells.size(), "Non-creature haunt should emit one SPELL: " + card.abilities());
        assertEquals(2, triggered.size(), "Non-creature haunt should emit two TRIGGERED: " + card.abilities());
        assertTrue(spells.get(0).descriptionText().contains("destroy target nonwhite"));
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().equals("haunt")),
                "One TRIGGERED should be the haunt keyword line: " + triggered);
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("haunts dies")
                && a.descriptionText().contains("destroy target nonwhite")),
                "One TRIGGERED should be the haunted-dies trigger with effect: " + triggered);
    }

    @Test
    void hauntCreatureEmitsKeywordAndTrigger() {
        // Blind Hunter is a creature with haunt.
        // Expected: no SPELL, 2 TRIGGERED (haunt keyword + haunted-dies trigger), plus flying STATIC.
        CardFace card = cards.face("b/blind_hunter.txt");
        assertEquals(0, countOfType(card, AbilityType.SPELL),
                "Creature haunt should not emit a SPELL: " + card.abilities());
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(2, triggered.size(), "Creature haunt should emit two TRIGGERED: " + card.abilities());
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().equals("haunt")),
                "One TRIGGERED should be the haunt keyword line: " + triggered);
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("haunts dies")
                && a.descriptionText().contains("loses 2 life")),
                "One TRIGGERED should be the haunted-dies trigger with effect: " + triggered);
    }

    /** Each visit attraction yields exactly one TRIGGERED, led by "visit —", with no SPELL children. */
    private Ability assertSingleVisitTrigger(ScriptText attraction) {
        CardFace card = cards.face(attraction.file());
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), attraction.file() + " should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertTrue(desc.contains(attraction.text()), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
        return visit;
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "Bumper Cars,   b/bumper_cars.txt,    must be blocked",
            "Ferris Wheel,  f/ferris_wheel.txt,   phases out",
            "Swinging Ship, s/swinging_ship.txt,  additional combat phase",
    })
    void visitWithDuplicatePrefixEmitsOneTrigger(@AggregateWith(ScriptText.FromNamedCsv.class) ScriptText attraction) {
        // SpellDescription starts with "Visit — " → Forge TrigDesc doubles it to "Visit — Visit — …"
        Ability visit = assertSingleVisitTrigger(attraction);
        assertFalse(visit.descriptionText().toLowerCase().contains("visit — visit"),
                "Description must not double 'visit —': " + visit.descriptionText());
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "Storybook Ride, s/storybook_ride.txt, exile the top",
            "Trash Bin,      t/trash_bin.txt,      mill",
    })
    void visitWithDuplicateSpellChildrenEmitsOneTrigger(@AggregateWith(ScriptText.FromNamedCsv.class) ScriptText attraction) {
        // SpellDescription doesn't start with "Visit — " → no prefix doubling, but
        // SpellEffect.fromChain produces spurious spell[1]/spell[2] sub-abilities.
        // The leading "visit —" also rules out the raw "visit:trigexile" key leaking through.
        assertSingleVisitTrigger(attraction);
    }

    @Test
    void tributeNotPaidTriggerIsIncluded() {
        // Pharagax Giant: K:Tribute:2 + SVar:TrigNotTribute (fires when tribute wasn't paid).
        // The keyword outputs "tribute 2"; the TrigNotTribute trigger must also be emitted
        // as a separate TRIGGERED ability — it is distinct oracle text not covered by the keyword.
        CardFace card = cards.face("p/pharagax_giant.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(2, triggered.size(),
                "Should have 2 TRIGGERED: keyword tribute + TrigNotTribute: " + card.abilities());
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("tribute")),
                "One TRIGGERED should be the tribute keyword: " + triggered);
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("tribute wasn't paid")),
                "One TRIGGERED should be the 'if tribute wasn't paid' effect: " + triggered);
    }

    @Test
    void visitCharmExpandsToOptionSubAbilities() {
        // Balloon Stand: Visit trigger whose overriding ability is a Charm SA.
        // Expected: 1 TRIGGERED with "visit" in header and 2 OPTION sub-abilities,
        // not embedded bullet text inside the triggered description.
        CardFace card = cards.face("b/balloon_stand.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(),
                "Balloon Stand should have 1 TRIGGERED ability: " + card.abilities());
        Ability visit = triggered.get(0);
        assertTrue(visit.descriptionText().toLowerCase().contains("visit"),
                "Triggered ability should have 'visit' in description: " + visit.descriptionText());
        List<Ability> options = visit.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(2, options.size(),
                "Visit charm should expand to 2 OPTION sub-abilities: " + visit.subAbilities());
        assertTrue(options.stream().anyMatch(a -> a.descriptionText().toLowerCase().contains("balloon")),
                "Should have a Balloon option: " + options);
    }

    @Test
    void mayEffectFromOpeningHandEmitsTriggeredAbility() {
        // Chancellor of the Tangle: MayEffectFromOpeningHand:ManaOnMain where ManaOnMain
        // has no SpellDescription; description is on the RevealCard SVar fallback.
        CardFace card = cards.face("c/chancellor_of_the_tangle.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertTrue(triggered.stream().anyMatch(a ->
                        a.descriptionText().toLowerCase().contains("may reveal")),
                "Should have a TRIGGERED ability containing 'may reveal': " + triggered);
    }

    @Test
    void wardKeywordClassifiedAsStatic() {
        // calim_djinn_emperor: K:Ward:2 should emit "static: ward {2}", not "triggered: ward {2}".
        CardFace card = cards.face("c/calim_djinn_emperor.txt");
        var statics = abilitiesOfType(card, AbilityType.STATIC);
        assertTrue(statics.stream().anyMatch(a -> a.descriptionText().toLowerCase().contains("ward")),
                "Ward should be classified as STATIC: " + card.abilities());
        assertEquals(0, abilitiesOfType(card, AbilityType.TRIGGERED).stream()
                        .filter(a -> a.descriptionText().toLowerCase().contains("ward")).count(),
                "Ward must NOT appear as TRIGGERED: " + card.abilities());
    }

    @Test
    void multiProtectionKeywordsEachEmitSeparateStaticLine() {
        // Elite Inquisitor has K:Protection:Vampire / K:Protection:Werewolf / K:Protection:Zombie.
        // Oracle groups them as "Protection from Vampires, from Werewolves, and from Zombies"
        // (one line), but the converter correctly models each Forge keyword as its own static
        // ability.  Pin this behaviour so it is not accidentally collapsed.
        var statics = abilitiesOfType(cards.face("e/elite_inquisitor.txt"), AbilityType.STATIC);
        long protections = statics.stream()
                .filter(a -> a.descriptionText().toLowerCase().contains("protection from"))
                .count();
        assertEquals(3, protections,
                "Each of the three protection keywords must produce its own static line: " + statics);
    }

    @Test
    void multiHexproofKeywordsEachEmitSeparateStaticLine() {
        // Jaheira, Harper Emissary has K:Hexproof:Artifact / K:Hexproof:Enchantment.
        // Oracle groups them as "Hexproof from artifacts and enchantments" (one line), but the
        // converter correctly models each Forge keyword as its own static ability.
        var statics = abilitiesOfType(cards.face("j/jaheira_harper_emissary.txt"), AbilityType.STATIC);
        long hexproofs = statics.stream()
                .filter(a -> a.descriptionText().toLowerCase().contains("hexproof from"))
                .count();
        assertEquals(2, hexproofs,
                "Each of the two hexproof keywords must produce its own static line: " + statics);
    }
}
