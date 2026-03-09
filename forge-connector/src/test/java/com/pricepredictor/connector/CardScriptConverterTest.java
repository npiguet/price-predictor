package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for CardScriptConverter using Forge's CardRules.Reader.
 * Requires the full Forge transitive dependency chain on the classpath.
 * Run via: mvn test -Pintegration
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class CardScriptConverterTest {

    private static final Path CARDS_FOLDER = findCardsFolder();

    private static Path findCardsFolder() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("forge").resolve("forge-gui").resolve("res").resolve("cardsfolder");
            if (Files.isDirectory(candidate)) return candidate;
            dir = dir.getParent();
        }
        throw new IllegalStateException(
                "Could not find forge/forge-gui/res/cardsfolder in any parent of " + Path.of("").toAbsolutePath());
    }

    private final CardScriptConverter converter = new CardScriptConverter();

    private MultiCard convert(String... lines) {
        return converter.convertCard(Arrays.asList(lines), "test.txt");
    }

    /** Load a real card script from the Forge cardsfolder by relative path. */
    private MultiCard convertFromFile(String relativePath) {
        try {
            Path file = CARDS_FOLDER.resolve(relativePath);
            List<String> lines = Files.readAllLines(file);
            return converter.convertCard(lines, file.getFileName().toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    // --- US1: Basic card conversion ---

    @Test
    void vanillaCreature() {
        MultiCard result = convertFromFile("g/grizzly_bears.txt");
        ConvertedCard card = result.faces().get(0);
        assertEquals("grizzly bears", card.name());
        assertTrue(card.manaCost().contains("{G}"));
        assertEquals("creature bear", card.types());
        assertEquals("2/2", card.powerToughness());
        assertTrue(card.abilities().isEmpty());
        assertNull(card.loyalty());
    }

    @Test
    void passiveKeywords() {
        MultiCard result = convertFromFile("s/serra_angel.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        assertEquals(2, keywords.size());
        assertEquals("keyword: flying", keywords.get(0).formatLine());
        assertEquals("keyword: vigilance", keywords.get(1).formatLine());
        assertNull(keywords.get(0).actionNumber());
    }

    @Test
    void activatedAbility() {
        MultiCard result = convertFromFile("l/llanowar_elves.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> activated = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ACTIVATED).toList();
        assertEquals(1, activated.size());
        String line = activated.get(0).formatLine();
        assertTrue(line.startsWith("activated[1]:"), "Expected activated[1]: but got: " + line);
        assertTrue(line.contains("{T}"), "Expected {T} in: " + line);
        assertTrue(line.contains("add {G}"), "Expected 'add {G}' in: " + line);
    }

    @Test
    void triggeredAbility() {
        MultiCard result = convertFromFile("t/thragtusk.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> triggered = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.TRIGGERED).toList();
        assertTrue(triggered.size() >= 1, "Expected at least 1 triggered ability");
        // First trigger: enters the battlefield, gain life
        String line = triggered.get(0).formatLine();
        assertTrue(line.startsWith("triggered:"));
        assertTrue(line.contains("CARDNAME"));
        assertTrue(line.contains("gain 5 life"));
    }

    @Test
    void staticAbility() {
        MultiCard result = convertFromFile("b/blood_moon.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> statics = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).toList();
        assertEquals(1, statics.size());
        assertTrue(statics.get(0).formatLine().contains("nonbasic lands are mountains"));
    }

    @Test
    void replacementEffect() {
        MultiCard result = convertFromFile("r/rest_in_peace.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> replacements = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.REPLACEMENT).toList();
        assertTrue(replacements.size() >= 1, "Expected at least 1 replacement effect");
        assertTrue(replacements.stream().anyMatch(r -> r.formatLine().contains("exile it instead")),
                "Expected replacement with 'exile it instead' but got: " + replacements);
    }

    @Test
    void engineMetadataExcluded() {
        MultiCard result = convert(
                "Name:Test Card",
                "ManaCost:1",
                "Types:Creature Human",
                "PT:1/1",
                "AI:RemoveDeck:Random",
                "DeckHints:Ability$LifeGain",
                "SVar:AnotherSVar:DB$ Something",
                "Oracle:");
        String output = OutputFormatter.formatMultiCard(result);
        assertFalse(output.contains("AI:"));
        assertFalse(output.contains("DeckHints"));
        assertFalse(output.contains("SVar"));
    }

    @Test
    void actionCounterIncrementsAcrossMixedTypes() {
        MultiCard result = convert(
                "Name:Versatile Card",
                "ManaCost:2 W",
                "Types:Creature Human",
                "PT:2/2",
                "K:Cycling:2",
                "A:AB$ GainLife | Cost$ T | LifeAmount$ 1 | SpellDescription$ Gain 1 life.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> actionable = card.abilities().stream()
                .filter(a -> a.actionNumber() != null).toList();
        assertTrue(actionable.size() >= 2, "Expected at least 2 actionable abilities");
        // Action numbers should be sequential
        for (int i = 0; i < actionable.size() - 1; i++) {
            assertTrue(actionable.get(i).actionNumber() < actionable.get(i + 1).actionNumber());
        }
    }

    @Test
    void textCasingCorrect() {
        MultiCard result = convertFromFile("l/llanowar_elves.txt");
        String output = OutputFormatter.formatMultiCard(result);
        assertTrue(output.contains("llanowar elves"), "Name should be lowercase");
        assertTrue(output.contains("{G}"), "Brace symbols should be uppercase");
        assertTrue(output.contains("{T}"), "Tap symbol should be uppercase");
    }

    @Test
    void variableXStaysUppercase() {
        // Standalone X in various positions
        assertEquals("-X: deal X damage", CardScriptConverter.applyTextCasing("-X: Deal X damage"));
        assertEquals("+X/+0 until end of turn", CardScriptConverter.applyTextCasing("+X/+0 until end of turn"));
        assertEquals("where X is the number", CardScriptConverter.applyTextCasing("Where X is the number"));

        // X inside words must stay lowercase
        assertEquals("exile target creature", CardScriptConverter.applyTextCasing("Exile target creature"));
        assertEquals("next end step", CardScriptConverter.applyTextCasing("Next end step"));
        assertEquals("tax each opponent", CardScriptConverter.applyTextCasing("Tax each opponent"));

        // X inside braces handled separately
        assertEquals("{X}{R}", CardScriptConverter.applyTextCasing("{X}{R}"));

        // Mixed: brace X and standalone X in same text
        assertEquals("pay {X}, where X is the number of counters",
                CardScriptConverter.applyTextCasing("Pay {X}, where X is the number of counters"));
    }

    @Test
    void noCostCardOmitsManaCostLine() {
        MultiCard result = convertFromFile("a/ancestral_vision.txt");
        ConvertedCard card = result.faces().get(0);
        assertNull(card.manaCost());
        String output = OutputFormatter.formatCard(card);
        assertFalse(output.contains("mana cost:"));
    }

    @Test
    void textPropertyIncluded() {
        MultiCard result = convert(
                "Name:Test Card",
                "ManaCost:1",
                "Types:Creature Human",
                "PT:1/1",
                "Text:This is flavor text.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        assertNotNull(card.text());
        String output = OutputFormatter.formatCard(card);
        assertTrue(output.contains("text: this is flavor text."));
    }

    // --- US2: Complex card types ---

    @Test
    void planeswalkerAbilities() {
        MultiCard result = convertFromFile("j/jace_beleren.txt");
        ConvertedCard card = result.faces().get(0);
        assertEquals("3", card.loyalty());
        List<AbilityLine> pwAbilities = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.PLANESWALKER).toList();
        assertEquals(3, pwAbilities.size());
        assertTrue(pwAbilities.get(0).formatLine().contains("[+2]:"), "Expected [+2]: but got: " + pwAbilities.get(0).formatLine());
        assertTrue(pwAbilities.get(1).formatLine().contains("[-1]:"), "Expected [-1]: but got: " + pwAbilities.get(1).formatLine());
        assertTrue(pwAbilities.get(2).formatLine().contains("[-10]:"), "Expected [-10]: but got: " + pwAbilities.get(2).formatLine());
        assertEquals(1, (int) pwAbilities.get(0).actionNumber());
        assertEquals(2, (int) pwAbilities.get(1).actionNumber());
        assertEquals(3, (int) pwAbilities.get(2).actionNumber());
    }

    @Test
    void sagaChapterAbilities() {
        MultiCard result = convertFromFile("t/the_eldest_reborn.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> chapters = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.CHAPTER).toList();
        assertEquals(3, chapters.size(), "Expected 3 chapter abilities but got: " + card.abilities());
        String ch1 = chapters.get(0).formatLine();
        String ch2 = chapters.get(1).formatLine();
        String ch3 = chapters.get(2).formatLine();
        assertTrue(ch1.startsWith("chapter:"), "Expected chapter: prefix but got: " + ch1);
        assertTrue(ch1.contains("I \u2014"), "Expected 'I \u2014' in: " + ch1);
        assertTrue(ch2.contains("II \u2014"), "Expected 'II \u2014' in: " + ch2);
        assertTrue(ch3.contains("III \u2014"), "Expected 'III \u2014' in: " + ch3);
        assertTrue(ch1.contains("sacrifices"), "Expected sacrifice text in: " + ch1);
        assertNull(chapters.get(0).actionNumber(), "Chapters should not have action numbers");
    }

    @Test
    void battleCardWithDefense() {
        MultiCard result = convertFromFile("i/invasion_of_kamigawa_rooftop_saboteurs.txt");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());

        // Front face: Battle with defense
        ConvertedCard front = result.faces().get(0);
        assertEquals("invasion of kamigawa", front.name());
        assertEquals("4", front.defense());
        assertNull(front.powerToughness());
        assertNull(front.loyalty());
        String frontOutput = OutputFormatter.formatCard(front);
        assertTrue(frontOutput.contains("defense: 4"), "Expected defense line in output");

        // Back face: creature, no defense
        ConvertedCard back = result.faces().get(1);
        assertEquals("rooftop saboteurs", back.name());
        assertNull(back.defense());
        assertEquals("2/3", back.powerToughness());
    }

    @Test
    void transformCard() {
        MultiCard result = convertFromFile("d/daring_sleuth_bearer_of_overwhelming_truths.txt");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());
        assertEquals("daring sleuth", result.faces().get(0).name());
        assertEquals("bearer of overwhelming truths", result.faces().get(1).name());
    }

    @Test
    void spellEffect() {
        MultiCard result = convertFromFile("l/lightning_bolt.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("CARDNAME deals 3 damage"));
        assertEquals(1, (int) spells.get(0).actionNumber());
        // No additional cost for a simple spell
        long additionalCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).count();
        assertEquals(0, additionalCosts, "Lightning Bolt should have no additional cost");
    }

    @Test
    void spellWithAdditionalCost() {
        MultiCard result = convertFromFile("a/abandon_hope.txt");
        ConvertedCard card = result.faces().get(0);

        // Additional cost on its own line
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.startsWith("additional cost:"),
                "Expected 'additional cost:' prefix but got: " + costLine);
        assertTrue(costLine.contains("discard X"),
                "Expected discard text in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // Spell effect on its own line
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line");
        assertTrue(spells.get(0).formatLine().contains("look at target opponent's hand"),
                "Expected spell text in: " + spells.get(0).formatLine());
    }

    @Test
    void cleaveEmittedAsAlternateCost() {
        // Alchemist's Gambit: Cleave is a NonBasicSpell alternate cost
        MultiCard result = convertFromFile("a/alchemists_gambit.txt");
        ConvertedCard card = result.faces().get(0);

        // Cleave should appear as alternate cost
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost line but got: " + card.abilities());
        String altLine = altCosts.get(0).formatLine();
        assertTrue(altLine.contains("cleave"), "Expected 'cleave' in: " + altLine);
        assertTrue(altLine.contains("{4}{U}{U}{R}"), "Expected '{4}{U}{U}{R}' in: " + altLine);
        // No buggy trailing characters
        assertFalse(altLine.matches(".*\\{R\\}\\d.*"), "No trailing digits after cost in: " + altLine);

        // Real spell lines should still have correct action numbers
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertTrue(spells.size() >= 1, "Expected at least 1 spell line but got: " + spells);
        List<Integer> actionNumbers = spells.stream()
                .map(AbilityLine::actionNumber).toList();
        for (int i = 0; i < actionNumbers.size() - 1; i++) {
            assertEquals(actionNumbers.get(i) + 1, (int) actionNumbers.get(i + 1),
                    "Action numbers should be sequential: " + actionNumbers);
        }
    }

    @Test
    void creatureWithAdditionalCost() {
        MultiCard result = convertFromFile("a/abhorrent_oculus.txt");
        ConvertedCard card = result.faces().get(0);

        // Additional cost should appear even without SpellDescription
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        assertTrue(costs.get(0).formatLine().contains("exile"),
                "Expected exile text in: " + costs.get(0).formatLine());
    }

    @Test
    void additionalCostFromRaiseCostStatic() {
        MultiCard result = convertFromFile("a/aether_tide.txt");
        ConvertedCard card = result.faces().get(0);

        // RaiseCost static should become additional cost, not static
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("discard X creature cards"),
                "Expected discard text in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // No static lines should remain
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, staticCount, "RaiseCost should not appear as static");

        // Spell effect on its own line
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line");
        assertTrue(spells.get(0).formatLine().contains("return X target creatures"),
                "Expected spell text in: " + spells.get(0).formatLine());

        // Additional cost should appear before spell in ability list
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void raiseCostOnOtherSpellsRemainsStatic() {
        // Aura of Silence: taxes opponent artifact/enchantment spells, not a self-cost
        MultiCard result = convertFromFile("a/aura_of_silence.txt");
        ConvertedCard card = result.faces().get(0);

        // Should be static, NOT additional cost
        List<AbilityLine> statics = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).toList();
        assertEquals(1, statics.size(), "RaiseCost on other spells should remain static");
        assertTrue(statics.get(0).description().contains("cost {2} more to cast"));

        long addCostCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).count();
        assertEquals(0, addCostCount, "Should have no additional cost lines");
    }

    @Test
    void optionalAdditionalCost() {
        MultiCard result = convertFromFile("a/analyze_the_pollen.txt");
        ConvertedCard card = result.faces().get(0);

        // OptionalCost static should become additional cost
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("you may collect evidence 8"),
                "Expected 'you may' description in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // No static lines should remain
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, staticCount, "OptionalCost should not appear as static");

        // Additional cost should appear before spell
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void alternateAdditionalCost() {
        MultiCard result = convertFromFile("a/annihilating_glare.txt");
        ConvertedCard card = result.faces().get(0);

        // AlternateAdditionalCost keyword should become additional cost line
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("sacrifice"), "Expected sacrifice in: " + costLine);
        assertTrue(costLine.contains("or"), "Expected 'or' joining alternatives in: " + costLine);

        // No keyword lines should remain for this
        long keywordCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE).count();
        assertEquals(0, keywordCount, "AlternateAdditionalCost should not appear as keyword");

        // Spell effect present
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("destroy target creature or planeswalker"));

        // Additional cost before spell
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void pawprintCharmOptions() {
        MultiCard result = convertFromFile("s/season_of_the_burrow.txt");
        ConvertedCard card = result.faces().get(0);

        // Charm generates a spell line from the combined choice descriptions
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line but got: " + card.abilities());

        List<AbilityLine> options = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(3, options.size(), "Expected 3 options but got: " + card.abilities());

        // Each option should have paw print cost prefix
        String o1 = options.get(0).formatLine();
        assertTrue(o1.contains("{P} \u2014"), "Expected {P} cost in: " + o1);
        assertTrue(o1.contains("create a 1/1"), "Expected token text in: " + o1);

        String o2 = options.get(1).formatLine();
        assertTrue(o2.contains("{P}{P} \u2014"), "Expected {P}{P} cost in: " + o2);
        assertTrue(o2.contains("exile target"), "Expected exile text in: " + o2);

        String o3 = options.get(2).formatLine();
        assertTrue(o3.contains("{P}{P}{P} \u2014"), "Expected {P}{P}{P} cost in: " + o3);
        assertTrue(o3.contains("return target"), "Expected reanimate text in: " + o3);
    }

    @Test
    void classEnchantmentLevels() {
        MultiCard result = convertFromFile("a/artificer_class.txt");
        ConvertedCard card = result.faces().get(0);
        assertEquals("enchantment class", card.types());

        List<AbilityLine> levels = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.LEVEL).toList();
        assertEquals(3, levels.size(), "Expected 3 level abilities but got: " + card.abilities());

        // Level 1: no cost prefix
        String l1 = levels.get(0).formatLine();
        assertTrue(l1.startsWith("level[1]:"), "Expected level[1]: but got: " + l1);
        assertTrue(l1.contains("first artifact spell"), "Expected level 1 text in: " + l1);
        assertFalse(l1.contains("{1}{U}:"), "Level 1 should not have a cost prefix");

        // Level 2: cost prefix
        String l2 = levels.get(1).formatLine();
        assertTrue(l2.startsWith("level[2]:"), "Expected level[2]: but got: " + l2);
        assertTrue(l2.contains("{1}{U}:"), "Expected cost {1}{U}: in: " + l2);
        assertTrue(l2.contains("reveal cards"), "Expected level 2 text in: " + l2);

        // Level 3: cost prefix
        String l3 = levels.get(2).formatLine();
        assertTrue(l3.startsWith("level[3]:"), "Expected level[3]: but got: " + l3);
        assertTrue(l3.contains("{5}{U}:"), "Expected cost {5}{U}: in: " + l3);
        assertTrue(l3.contains("create a token"), "Expected level 3 text in: " + l3);

        // No activated or static lines should remain
        long activatedCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ACTIVATED).count();
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, activatedCount, "No activated lines expected for Class card");
        assertEquals(0, staticCount, "No static lines expected for Class card");
    }

    @Test
    void classWithEtbReplacementLevel1() {
        MultiCard result = convertFromFile("b/bard_class.txt");
        ConvertedCard card = result.faces().get(0);

        List<AbilityLine> levels = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.LEVEL).toList();
        assertEquals(3, levels.size(), "Expected 3 level abilities but got: " + card.abilities());

        // Level 1 comes from ETBReplacement (not a static)
        String l1 = levels.get(0).formatLine();
        assertTrue(l1.startsWith("level[1]:"), "Expected level[1]: but got: " + l1);
        assertTrue(l1.contains("legendary creatures"), "Expected level 1 text in: " + l1);

        // No raw ETBReplacement keyword lines
        long rawKeywords = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE)
                        && a.description().contains("etbreplacement")).count();
        assertEquals(0, rawKeywords, "ETBReplacement should not appear as raw keyword");
    }

    @Test
    void etbReplacementOnNonClassCard() {
        MultiCard result = convertFromFile("f/flickering_ward.txt");
        ConvertedCard card = result.faces().get(0);

        // ETBReplacement should be a replacement line, not a raw keyword
        long rawKeywords = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE)
                        && a.description().contains("etbreplacement")).count();
        assertEquals(0, rawKeywords, "ETBReplacement should not appear as raw keyword");

        List<AbilityLine> replacements = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.REPLACEMENT).toList();
        assertTrue(replacements.stream().anyMatch(r -> r.description().contains("choose a color")),
                "Expected 'choose a color' replacement but got: " + replacements);
    }

    @Test
    void companionKeywordIncludesRestriction() {
        MultiCard result = convertFromFile("g/gyruda_doom_of_depths.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        assertTrue(keywords.stream().anyMatch(k ->
                        k.description().contains("companion") && k.description().contains("even mana value")),
                "Expected companion with deck restriction but got: " + keywords);
    }

    @Test
    void doubleKickerIncludesBothCosts() {
        MultiCard result = convertFromFile("a/archangel_of_wrath.txt");
        ConvertedCard card = result.faces().get(0);
        // Kicker should be reclassified as additional cost with both costs
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("kicker"), "Expected 'kicker' in: " + costLine);
        assertTrue(costLine.contains("{B}"), "Expected '{B}' in: " + costLine);
        assertTrue(costLine.contains("{R}"), "Expected '{R}' in: " + costLine);
        assertTrue(costLine.contains("and/or"), "Expected 'and/or' in: " + costLine);
    }

    @Test
    void giftKeywordIncludesParameter() {
        MultiCard result = convertFromFile("d/dawns_truce.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        // Gift should include its parameter
        assertTrue(keywords.stream().anyMatch(k -> k.description().contains("gift a card")),
                "Expected 'gift a card' keyword but got: " + keywords);
    }

    // --- Alternate cost keyword classification ---

    @Test
    void ninjutsuClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("y/yuffie_materia_hunter.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Ninjutsu but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("ninjutsu"),
                "Expected 'ninjutsu' in: " + altCosts.get(0).description());
        assertTrue(altCosts.get(0).description().contains("{1}{R}"),
                "Expected '{1}{R}' in: " + altCosts.get(0).description());
    }

    @Test
    void embalmClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("v/vizier_of_many_faces.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Embalm but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("embalm"),
                "Expected 'embalm' in: " + altCosts.get(0).description());
        assertTrue(altCosts.get(0).description().contains("{3}{U}{U}"),
                "Expected '{3}{U}{U}' in: " + altCosts.get(0).description());
    }

    @Test
    void miracleClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("z/zephyrim.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Miracle but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("miracle"),
                "Expected 'miracle' in: " + altCosts.get(0).description());
        // Squad should be additional cost, not alternate
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Squad");
    }

    @Test
    void unearthClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("y/yotian_frontliner.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Unearth but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("unearth"),
                "Expected 'unearth' in: " + altCosts.get(0).description());
    }

    @Test
    void awakenClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("s/sheer_drop.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Awaken but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("awaken"),
                "Expected 'awaken' in: " + altCosts.get(0).description());
    }

    @Test
    void suspendClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("w/wheel_of_fate.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Suspend but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("suspend"),
                "Expected 'suspend' in: " + altCosts.get(0).description());
    }

    @Test
    void jumpStartClassifiedAsAlternateCost() {
        MultiCard result = convertFromFile("s/surge_of_acclaim.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Jump-start but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("jump-start"),
                "Expected 'jump-start' in: " + altCosts.get(0).description());
    }

    // --- Additional cost keyword classification ---

    @Test
    void bargainClassifiedAsAdditionalCost() {
        MultiCard result = convertFromFile("t/troublemaker_ouphe.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Bargain but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("bargain"),
                "Expected 'bargain' in: " + addCosts.get(0).description());
    }

    @Test
    void conspireClassifiedAsAdditionalCost() {
        MultiCard result = convertFromFile("t/traitors_roar.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Conspire but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("conspire"),
                "Expected 'conspire' in: " + addCosts.get(0).description());
    }

    @Test
    void spliceClassifiedAsAdditionalCost() {
        MultiCard result = convertFromFile("w/wear_away.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Splice but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("splice"),
                "Expected 'splice' in: " + addCosts.get(0).description());
    }

    @Test
    void spreeClassifiedAsAdditionalCost() {
        MultiCard result = convertFromFile("u/unfortunate_accident.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Spree but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("spree"),
                "Expected 'spree' in: " + addCosts.get(0).description());
    }

    // --- Cost reduction keyword classification ---

    @Test
    void affinityClassifiedAsCostReduction() {
        MultiCard result = convertFromFile("s/sojourners_companion.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Affinity but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("affinity"),
                "Expected 'affinity' in: " + reductions.get(0).description());
        assertEquals("cost reduction: affinity for artifacts", reductions.get(0).formatLine());
    }

    @Test
    void delveClassifiedAsCostReduction() {
        MultiCard result = convertFromFile("w/will_of_the_naga.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Delve but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("delve"),
                "Expected 'delve' in: " + reductions.get(0).description());
    }

    @Test
    void improviseClassifiedAsCostReduction() {
        MultiCard result = convertFromFile("w/whir_of_invention.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Improvise but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("improvise"),
                "Expected 'improvise' in: " + reductions.get(0).description());
    }

    @Test
    void convokeClassifiedAsCostReduction() {
        MultiCard result = convert(
                "Name:Test Convoke Spell",
                "ManaCost:3 W W",
                "Types:Sorcery",
                "K:Convoke",
                "A:SP$ Destroy | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Convoke but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("convoke"),
                "Expected 'convoke' in: " + reductions.get(0).description());
    }

    // --- Cost lines sort to top ---

    @Test
    void costReductionSortsToTop() {
        MultiCard result = convertFromFile("w/whir_of_invention.txt");
        ConvertedCard card = result.faces().get(0);
        // Cost reduction should come before spell abilities
        List<AbilityLine> abilities = card.abilities();
        assertTrue(abilities.size() >= 2, "Expected at least 2 abilities");
        assertEquals(AbilityType.COST_REDUCTION, abilities.get(0).type(),
                "Cost reduction should be first ability but got: " + abilities);
    }

    @Test
    void allCostTypesSortBeforeOtherAbilities() {
        // Card with alternate cost, additional cost, cost reduction, and a spell
        MultiCard result = convertFromFile("z/zephyrim.txt");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> abilities = card.abilities();
        // Find the first non-cost ability
        int firstNonCostIdx = -1;
        int lastCostIdx = -1;
        for (int i = 0; i < abilities.size(); i++) {
            AbilityType t = abilities.get(i).type();
            if (t == AbilityType.ALTERNATE_COST || t == AbilityType.ADDITIONAL_COST
                    || t == AbilityType.COST_REDUCTION) {
                lastCostIdx = i;
            } else if (firstNonCostIdx == -1) {
                firstNonCostIdx = i;
            }
        }
        if (lastCostIdx >= 0 && firstNonCostIdx >= 0) {
            assertTrue(lastCostIdx < firstNonCostIdx,
                    "All cost lines should appear before non-cost lines but got: " + abilities);
        }
    }

    @Test
    void layoutDetection() {
        // Transform
        MultiCard transform = convert("Name:A", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AlternateMode:DoubleFaced", "Oracle:", "ALTERNATE",
                "Name:B", "Types:Creature Werewolf", "PT:2/2", "Oracle:");
        assertEquals("transform", transform.layout());

        // Split
        MultiCard split = convert("Name:Fire", "ManaCost:1 R", "Types:Instant",
                "AlternateMode:Split", "Oracle:", "ALTERNATE",
                "Name:Ice", "ManaCost:1 U", "Types:Instant", "Oracle:");
        assertEquals("split", split.layout());

        // Adventure
        MultiCard adv = convert("Name:Bonecrusher Giant", "ManaCost:2 R", "Types:Creature Giant",
                "PT:4/3", "AlternateMode:Adventure", "Oracle:", "ALTERNATE",
                "Name:Stomp", "ManaCost:1 R", "Types:Instant Adventure", "Oracle:");
        assertEquals("adventure", adv.layout());

        // Meld (half without back face)
        MultiCard meld = convert("Name:The Mightstone and Weakstone", "ManaCost:5",
                "Types:Legendary Artifact Powerstone",
                "MeldPair:Urza, Lord Protector", "AlternateMode:Meld", "Oracle:");
        assertEquals("meld", meld.layout());
    }

    // --- Meld-half cards (no inline back face) ---

    @Test
    void meldHalf_mightstoneAndWeakstone_doesNotThrow() {
        MultiCard result = convertFromFile("t/the_mightstone_and_weakstone.txt");
        assertEquals("meld", result.layout());
        assertEquals(1, result.faces().size(), "Meld-half should have only the front face");
        assertEquals("the mightstone and weakstone", result.faces().get(0).name());
    }

    @Test
    void meldHalf_phyrexianDragonEngine_doesNotThrow() {
        MultiCard result = convertFromFile("p/phyrexian_dragon_engine.txt");
        assertEquals("meld", result.layout());
        assertEquals(1, result.faces().size(), "Meld-half should have only the front face");
        assertEquals("phyrexian dragon engine", result.faces().get(0).name());
        assertEquals("2/2", result.faces().get(0).powerToughness());
    }

    // --- Cards with Count$Valid SVars (previously caused NPE) ---

    @Test
    void bloomTender_doesNotThrow() {
        MultiCard result = convertFromFile("b/bloom_tender.txt");
        assertNotNull(result);
        assertEquals("bloom tender", result.faces().get(0).name());
        assertEquals("1/1", result.faces().get(0).powerToughness());
    }

    @Test
    void faeburrowElder_doesNotThrow() {
        MultiCard result = convertFromFile("f/faeburrow_elder.txt");
        assertNotNull(result);
        assertEquals("faeburrow elder", result.faces().get(0).name());
        assertEquals("0/0", result.faces().get(0).powerToughness());
    }

    // --- Charm: sub-ability description walking ---

    @Test
    void whatMustBeDone_charmWithSubAbilityDescription() {
        MultiCard result = convertFromFile("w/what_must_be_done.txt");
        assertNotNull(result);
        ConvertedCard card = result.faces().get(0);

        List<AbilityLine> options = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(2, options.size(), "Expected 2 options but got: " + card.abilities());

        String o1 = options.get(0).formatLine();
        assertTrue(o1.contains("destroy all artifacts and creatures"), "Expected destroy text in: " + o1);

        String o2 = options.get(1).formatLine();
        assertTrue(o2.contains("release juno"), "Expected 'release juno' in: " + o2);
        assertTrue(o2.contains("return target historic permanent card"), "Expected return text in: " + o2);
        // Reminder text should be stripped
        assertFalse(o2.contains("Artifacts, legendaries, and Sagas are historic"), "Reminder text should be stripped: " + o2);
    }

    @Test
    void charm_choiceWithNoDescriptionAnywhere_throws() {
        assertThrows(Exception.class, () -> convert(
                "Name:Bad Charm",
                "ManaCost:1 U",
                "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBNoop,DBDraw",
                "SVar:DBNoop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card.",
                "Oracle:"));
    }

    @Test
    void charm_choiceDescriptionOnSubAbility() {
        MultiCard result = convert(
                "Name:Chain Charm",
                "ManaCost:1 G",
                "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBChainTop,DBDirect",
                "SVar:DBChainTop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0 | SubAbility$ DBChainDesc",
                "SVar:DBChainDesc:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card from the chain.",
                "SVar:DBDirect:DB$ GainLife | LifeAmount$ 3 | SpellDescription$ Gain 3 life.",
                "Oracle:");
        assertNotNull(result);
        ConvertedCard card = result.faces().get(0);

        List<AbilityLine> options = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(2, options.size(), "Expected 2 options but got: " + card.abilities());

        String o1 = options.get(0).formatLine();
        assertTrue(o1.contains("draw a card from the chain"), "Expected sub-ability description in: " + o1);

        String o2 = options.get(1).formatLine();
        assertTrue(o2.contains("gain 3 life"), "Expected direct description in: " + o2);
    }

    @Test
    void tarnationVista_doesNotThrow() {
        MultiCard result = convertFromFile("t/tarnation_vista.txt");
        assertNotNull(result);
        assertEquals("tarnation vista", result.faces().get(0).name());
    }
}
