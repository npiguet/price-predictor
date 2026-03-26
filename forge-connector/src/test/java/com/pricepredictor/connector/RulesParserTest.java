package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
@ExtendWith(ForgeExtension.class)
class RulesParserTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser converter = new RulesParser();

    private MultiCard convert(String... lines) {
        return converter.parseScript(Arrays.asList(lines), "test.txt");
    }

    private MultiCard convertFromFile(String relativePath) {
        try {
            Path file = CARDS_FOLDER.resolve(relativePath);
            List<String> lines = Files.readAllLines(file);
            return converter.parseScript(lines, file.getFileName().toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private CardFace face(String relativePath) {
        return convertFromFile(relativePath).faces().get(0);
    }

    private List<Ability> abilitiesOfType(CardFace card, AbilityType type) {
        return card.abilities().stream().filter(a -> a.type() == type).toList();
    }

    private long countOfType(CardFace card, AbilityType type) {
        return card.abilities().stream().filter(a -> a.type() == type).count();
    }

    /** Collect all OPTION abilities — both top-level and nested as sub-abilities. */
    private List<Ability> allOptions(CardFace card) {
        List<Ability> options = new ArrayList<>();
        for (Ability a : card.abilities()) {
            if (a.type() == AbilityType.OPTION) {
                options.add(a);
            }
            for (Ability sub : a.subAbilities()) {
                if (sub.type() == AbilityType.OPTION) {
                    options.add(sub);
                }
            }
        }
        return options;
    }

    private void assertHasAbility(CardFace card, AbilityType type, String... containsTexts) {
        var abilities = abilitiesOfType(card, type);
        assertFalse(abilities.isEmpty(), "Expected at least one " + type + " but got: " + card.abilities());
        for (String text : containsTexts) {
            assertTrue(abilities.stream().anyMatch(a -> a.descriptionText().contains(text)),
                    "Expected '" + text + "' in " + type + " abilities but got: " + abilities);
        }
    }

    // --- US1: Basic card conversion ---

    @Test
    void vanillaCreature() {
        CardFace card = face("g/grizzly_bears.txt");
        assertEquals("grizzly bears", card.name());
        assertTrue(card.manaCost().contains("{G}"));
        assertEquals("creature bear", card.types());
        assertEquals("2/2", card.powerToughness());
        assertTrue(card.abilities().isEmpty());
        assertNull(card.loyalty());
    }

    @Test
    void passiveKeywords() {
        var keywords = abilitiesOfType(face("s/serra_angel.txt"), AbilityType.STATIC);
        assertTrue(keywords.stream().anyMatch(k -> k.descriptionText().equals("flying")));
        assertTrue(keywords.stream().anyMatch(k -> k.descriptionText().equals("vigilance")));
        assertFalse(keywords.get(0).formatLine().contains("["), "Static keywords should have no action number");
    }

    @Test
    void internalKeywordUsesReminderText() {
        // MayFlashSac is a Forge-internal keyword with a camelCase name.
        // The converter should use its reminder text as the description.
        CardFace card = face("l/lightning_reflexes.txt");
        assertTrue(card.abilities().stream().anyMatch(
                a -> a.descriptionText().contains("you may cast CARDNAME as though it had flash")),
                "MayFlashSac should use reminder text: " + card.abilities());
    }

    @Test
    void mayFlashCostUsesReminderText() {
        CardFace card = face("a/asinine_antics.txt");
        var costs = abilitiesOfType(card, AbilityType.ADDITIONAL_COST);
        assertEquals(1, costs.size());
        assertTrue(costs.get(0).descriptionText().contains("you may cast CARDNAME as though it had flash"),
                "MayFlashCost should use reminder text: " + costs.get(0).descriptionText());
    }

    @Test
    void protectionKeywordIncludesColor() {
        var statics = abilitiesOfType(face("a/animar_soul_of_elements.txt"), AbilityType.STATIC);
        assertTrue(statics.stream().anyMatch(k -> k.descriptionText().contains("protection from white")));
        assertTrue(statics.stream().anyMatch(k -> k.descriptionText().contains("protection from black")));
    }

    @Test
    void protectionKeywordTypeUsesFromFormat() {
        // K:Protection:Creature should emit "protection from creature", not the
        // internal Forge filter syntax "protection:creature".
        var statics = abilitiesOfType(face("b/beloved_chaplain.txt"), AbilityType.STATIC);
        assertTrue(statics.stream().anyMatch(k -> k.descriptionText().contains("protection from")),
                "Protection should use 'protection from X' format, not 'protection:X': " + statics);
    }

    @Test
    void protectionKeywordFilterExpressionUsesHumanReadableTarget() {
        // K:Protection:Card.MultiColor:multicolored should emit
        // "protection from multicolored", not "protection:card.multicolor:multicolored".
        var statics = abilitiesOfType(face("e/enemy_of_the_guildpact.txt"), AbilityType.STATIC);
        assertTrue(statics.stream().anyMatch(k -> k.descriptionText().contains("protection from multicolored")),
                "Filter-based protection should use human-readable target: " + statics);
    }

    @Test
    void activatedAbility() {
        CardFace card = face("l/llanowar_elves.txt");
        String output = card.formatText();
        assertTrue(output.contains("activated[1]:"), output);
        assertTrue(output.contains("{T}"), output);
        assertTrue(output.contains("add {G}"), output);
    }

    @Test
    void triggeredAbility() {
        var triggered = abilitiesOfType(face("t/thragtusk.txt"), AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        String line = triggered.get(0).formatLine();
        assertTrue(line.startsWith("triggered:"));
        assertTrue(line.contains("CARDNAME"));
        assertTrue(line.contains("gain 5 life"));
    }

    @Test
    void staticAbility() {
        var statics = abilitiesOfType(face("b/blood_moon.txt"), AbilityType.STATIC);
        assertEquals(1, statics.size());
        assertTrue(statics.get(0).formatLine().contains("nonbasic lands are mountains"));
    }

    @Test
    void replacementEffect() {
        var replacements = abilitiesOfType(face("r/rest_in_peace.txt"), AbilityType.REPLACEMENT);
        assertFalse(replacements.isEmpty());
        assertTrue(replacements.stream().anyMatch(r -> r.formatLine().contains("exile it instead")));
    }

    @Test
    void staticTriggerEmittedAsReplacement() {
        // Domesticated Mammoth has a trigger with Static$ True ("as this enters").
        // In MTG rules these are replacement effects, not triggered abilities.
        CardFace card = face("d/domesticated_mammoth.txt");
        var replacements = abilitiesOfType(card, AbilityType.REPLACEMENT);
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).descriptionText().contains("token copy of pacifism"));
        assertEquals(0, countOfType(card, AbilityType.TRIGGERED));
    }

    @Test
    void secondaryTriggerNotSkipped() {
        // Decorated Champion has a trigger marked Secondary$ True in Forge.
        // It should still be emitted since it's the card's actual ability text.
        CardFace card = face("d/decorated_champion.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size());
        assertTrue(triggered.get(0).descriptionText().contains("put a +1/+1 counter"));
    }

    @Test
    void engineMetadataExcluded() {
        String output = convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AI:RemoveDeck:Random", "DeckHints:Ability$LifeGain",
                "SVar:AnotherSVar:DB$ Something", "Oracle:").formatText();
        assertFalse(output.contains("AI:"));
        assertFalse(output.contains("DeckHints"));
        assertFalse(output.contains("SVar"));
    }

    @Test
    void actionCounterIncrementsAcrossMixedTypes() {
        CardFace card = convert(
                "Name:Versatile Card", "ManaCost:2 W", "Types:Creature Human", "PT:2/2",
                "K:Cycling:2", "A:AB$ GainLife | Cost$ T | LifeAmount$ 1 | SpellDescription$ Gain 1 life.",
                "Oracle:").faces().get(0);
        // Verify sequential action numbers in formatted output
        String output = card.formatText();
        assertTrue(output.contains("[1]:"), "Should have action number 1: " + output);
        assertTrue(output.contains("[2]:"), "Should have action number 2: " + output);
        assertTrue(output.indexOf("[1]:") < output.indexOf("[2]:"),
                "Action numbers should be sequential in output: " + output);
    }

    @Test
    void textCasingCorrect() {
        String output = convertFromFile("l/llanowar_elves.txt").formatText();
        assertTrue(output.contains("llanowar elves"));
        assertTrue(output.contains("{G}"));
        assertTrue(output.contains("{T}"));
    }

    @Test
    void variableXStaysUppercase() {
        // Standalone X preserved
        assertEquals("-X: deal X damage", AbilityDescription.applyCasing("-X: Deal X damage"));
        assertEquals("+X/+0 until end of turn", AbilityDescription.applyCasing("+X/+0 until end of turn"));
        assertEquals("where X is the number", AbilityDescription.applyCasing("Where X is the number"));
        // X inside words lowercased
        assertEquals("exile target creature", AbilityDescription.applyCasing("Exile target creature"));
        assertEquals("next end step", AbilityDescription.applyCasing("Next end step"));
        assertEquals("tax each opponent", AbilityDescription.applyCasing("Tax each opponent"));
        // Braces
        assertEquals("{X}{R}", AbilityDescription.applyCasing("{X}{R}"));
        // Mixed
        assertEquals("pay {X}, where X is the number of counters",
                AbilityDescription.applyCasing("Pay {X}, where X is the number of counters"));
    }

    @Test
    void noCostCardOmitsManaCostLine() {
        CardFace card = face("a/ancestral_vision.txt");
        assertNull(card.manaCost());
        assertFalse(card.formatText().contains("mana cost:"));
    }

    @Test
    void textPropertyIncluded() {
        String output = convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "Text:This is flavor text.", "Oracle:").faces().get(0).formatText();
        assertTrue(output.contains("text: this is flavor text."));
    }

    // --- US2: Complex card types ---

    @Test
    void planeswalkerAbilities() {
        CardFace card = face("j/jace_beleren.txt");
        assertEquals("3", card.loyalty());
        var pw = abilitiesOfType(card, AbilityType.PLANESWALKER);
        assertEquals(3, pw.size());
        assertTrue(pw.get(0).formatLine().contains("[+2]:"));
        assertTrue(pw.get(1).formatLine().contains("[-1]:"));
        assertTrue(pw.get(2).formatLine().contains("[-10]:"));
        // Verify sequential action numbers in formatted output
        String output = card.formatText();
        assertTrue(output.contains("planeswalker[1]:"));
        assertTrue(output.contains("planeswalker[2]:"));
        assertTrue(output.contains("planeswalker[3]:"));
    }

    @Test
    void sagaChapterAbilities() {
        var chapters = abilitiesOfType(face("t/the_eldest_reborn.txt"), AbilityType.CHAPTER);
        assertEquals(3, chapters.size());
        assertTrue(chapters.get(0).formatLine().startsWith("chapter:"));
        assertTrue(chapters.get(0).formatLine().contains("I \u2014"));
        assertTrue(chapters.get(1).formatLine().contains("II \u2014"));
        assertTrue(chapters.get(2).formatLine().contains("III \u2014"));
        assertTrue(chapters.get(0).formatLine().contains("sacrifices"));
        assertFalse(chapters.get(0).formatLine().contains("["), "Chapters should have no action number");
    }

    @Test
    void battleCardWithDefense() {
        MultiCard result = convertFromFile("i/invasion_of_kamigawa_rooftop_saboteurs.txt");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());

        CardFace front = result.faces().get(0);
        assertEquals("invasion of kamigawa", front.name());
        assertEquals("4", front.defense());
        assertNull(front.powerToughness());
        assertTrue(front.formatText().contains("defense: 4"));

        CardFace back = result.faces().get(1);
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
        CardFace card = face("l/lightning_bolt.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("CARDNAME deals 3 damage"));
        // Verify action number in formatted output
        String output = card.formatText();
        assertTrue(output.contains("spell[1]:"));
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void spellWithAdditionalCost() {
        CardFace card = face("a/abandon_hope.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.startsWith("additional cost:"));
        assertTrue(costLine.contains("discard X"));
        assertFalse(costLine.contains("as an additional cost to cast"));

        assertTrue(abilitiesOfType(card, AbilityType.SPELL).get(0).formatLine()
                .contains("look at target opponent's hand"));
    }

    @Test
    void cleaveEmittedAsAlternateCost() {
        CardFace card = face("a/alchemists_gambit.txt");

        String altLine = abilitiesOfType(card, AbilityType.ALTERNATE_COST).get(0).formatLine();
        assertTrue(altLine.contains("cleave"));
        assertTrue(altLine.contains("{4}{U}{U}{R}"));
        assertFalse(altLine.matches(".*\\{R}\\d.*"), "No trailing digits after cost");

        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty());
        // Verify sequential action numbering in formatted output
        String output = card.formatText();
        for (int i = 0; i < spells.size(); i++) {
            assertTrue(output.contains("spell[" + (i + 1) + "]:"),
                    "Expected spell[" + (i + 1) + "] in output: " + output);
        }
    }

    @Test
    void creatureWithAdditionalCost() {
        CardFace card = face("a/abhorrent_oculus.txt");
        var costs = abilitiesOfType(card, AbilityType.ADDITIONAL_COST);
        assertEquals(1, costs.size());
        assertTrue(costs.get(0).formatLine().contains("exile"));
    }

    @Test
    void additionalCostFromRaiseCostStatic() {
        CardFace card = face("a/aether_tide.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("discard X creature cards"));
        assertFalse(costLine.contains("as an additional cost to cast"));
        assertEquals(0, countOfType(card, AbilityType.STATIC), "RaiseCost should not appear as static");

        assertTrue(abilitiesOfType(card, AbilityType.SPELL).get(0).formatLine()
                .contains("return X target creatures"));
        assertCostsBeforeSpells(card);
    }

    @Test
    void raiseCostOnOtherSpellsRemainsStatic() {
        CardFace card = face("a/aura_of_silence.txt");
        assertEquals(1, abilitiesOfType(card, AbilityType.STATIC).size());
        assertTrue(abilitiesOfType(card, AbilityType.STATIC).get(0).descriptionText().contains("cost {2} more to cast"));
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void optionalAdditionalCost() {
        CardFace card = face("a/analyze_the_pollen.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("you may collect evidence 8"));
        assertFalse(costLine.contains("as an additional cost to cast"));
        assertEquals(0, countOfType(card, AbilityType.STATIC));
        assertCostsBeforeSpells(card);
    }

    @Test
    void alternateAdditionalCost() {
        CardFace card = face("a/annihilating_glare.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("sacrifice"));
        assertTrue(costLine.contains("or"));
        assertEquals(0, countOfType(card, AbilityType.STATIC)
                + countOfType(card, AbilityType.ACTIVATED),
                "AlternateAdditionalCost should not appear as keyword");

        assertTrue(abilitiesOfType(card, AbilityType.SPELL).get(0).formatLine()
                .contains("destroy target creature or planeswalker"));
        assertCostsBeforeSpells(card);
    }

    @Test
    void pawprintCharmOptions() {
        CardFace card = face("s/season_of_the_burrow.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());

        // Charm options are sub-abilities of the spell
        var options = spells.get(0).subAbilities();
        assertEquals(3, options.size());
        assertTrue(options.get(0).formatLine().contains("{P} \u2014"));
        assertTrue(options.get(0).formatLine().contains("create a 1/1"));
        assertTrue(options.get(1).formatLine().contains("{P}{P} \u2014"));
        assertTrue(options.get(1).formatLine().contains("exile target"));
        assertTrue(options.get(2).formatLine().contains("{P}{P}{P} \u2014"));
        assertTrue(options.get(2).formatLine().contains("return target"));
    }

    @Test
    void classEnchantmentLevels() {
        CardFace card = face("a/artificer_class.txt");
        assertEquals("enchantment class", card.types());

        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        // Class levels use ordinal() for fixed numbering in formatted output
        String output = card.formatText();
        assertTrue(output.contains("level[1]:"), "Level 1 in output: " + output);
        assertTrue(output.contains("level[2]:"), "Level 2 in output: " + output);
        assertTrue(output.contains("level[3]:"), "Level 3 in output: " + output);
        assertTrue(levels.get(0).descriptionText().contains("first artifact spell"));
        assertFalse(levels.get(0).descriptionText().contains("{1}{U}:"));
        assertTrue(levels.get(1).descriptionText().contains("{1}{U}:"));
        assertTrue(levels.get(1).descriptionText().contains("reveal cards"));
        assertTrue(levels.get(2).descriptionText().contains("{5}{U}:"));
        assertTrue(levels.get(2).descriptionText().contains("create a token"));

        assertEquals(0, countOfType(card, AbilityType.ACTIVATED));
        assertEquals(0, countOfType(card, AbilityType.STATIC));
    }

    @Test
    void classWithEtbReplacementLevel1() {
        CardFace card = face("b/bard_class.txt");
        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        String output = card.formatText();
        assertTrue(output.contains("level[1]:"), "Level 1 in output: " + output);
        assertTrue(levels.get(0).descriptionText().contains("legendary creatures"));
        assertNoRawEtbReplacementKeyword(card);
    }

    @Test
    void etbReplacementOnNonClassCard() {
        CardFace card = face("f/flickering_ward.txt");
        assertNoRawEtbReplacementKeyword(card);
        assertTrue(abilitiesOfType(card, AbilityType.REPLACEMENT).stream()
                .anyMatch(r -> r.descriptionText().contains("choose a color")));
    }

    @Test
    void etbCounterWithDescriptionFallback() {
        CardFace card = face("a/ambitious_dragonborn.txt");
        assertNoRawEtbReplacementKeyword(card);
        var replacements = abilitiesOfType(card, AbilityType.REPLACEMENT);
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).descriptionText().contains("CARDNAME enters with X +1/+1 counters"));
    }

    @Test
    void companionKeywordIncludesRestriction() {
        assertHasAbility(face("g/gyruda_doom_of_depths.txt"),
                AbilityType.STATIC, "companion", "even mana value");
    }

    @Test
    void doubleKickerIncludesBothCosts() {
        String costLine = abilitiesOfType(face("a/archangel_of_wrath.txt"),
                AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("kicker"));
        assertTrue(costLine.contains("{B}"));
        assertTrue(costLine.contains("{R}"));
        assertTrue(costLine.contains("and/or"));
    }

    @Test
    void giftKeywordIncludesParameter() {
        assertHasAbility(face("d/dawns_truce.txt"), AbilityType.STATIC, "gift a card");
    }

    // --- Keyword cost classification (parameterized) ---

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
    void keywordClassifiedAsAlternateCost(String file, String keyword) {
        assertHasAbility(face(file), AbilityType.ALTERNATE_COST, keyword);
    }

    @ParameterizedTest
    @CsvSource({
            "t/troublemaker_ouphe.txt,  bargain",
            "t/traitors_roar.txt,       conspire",
            "w/wear_away.txt,           splice",
            "u/unfortunate_accident.txt, spree",
    })
    void keywordClassifiedAsAdditionalCost(String file, String keyword) {
        assertHasAbility(face(file), AbilityType.ADDITIONAL_COST, keyword);
    }

    @ParameterizedTest
    @CsvSource({
            "s/sojourners_companion.txt, affinity",
            "w/will_of_the_naga.txt,    delve",
            "w/whir_of_invention.txt,   improvise",
    })
    void keywordClassifiedAsCostReduction(String file, String keyword) {
        assertHasAbility(face(file), AbilityType.COST_REDUCTION, keyword);
    }

    @Test
    void convokeClassifiedAsCostReduction() {
        CardFace card = convert(
                "Name:Test Convoke Spell", "ManaCost:3 W W", "Types:Sorcery",
                "K:Convoke", "A:SP$ Destroy | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "Oracle:").faces().get(0);
        assertHasAbility(card, AbilityType.COST_REDUCTION, "convoke");
    }

    // --- Cost ordering ---

    @Test
    void allCostTypesSortBeforeOtherAbilities() {
        // Whir of Invention: cost reduction should be first ability
        CardFace whir = face("w/whir_of_invention.txt");
        assertTrue(whir.abilities().size() >= 2);
        assertEquals(AbilityType.COST_REDUCTION, whir.abilities().get(0).type());

        // Zephyrim: alternate cost + additional cost + keywords
        CardFace zeph = face("z/zephyrim.txt");
        int firstNonCostIdx = -1, lastCostIdx = -1;
        for (int i = 0; i < zeph.abilities().size(); i++) {
            AbilityType t = zeph.abilities().get(i).type();
            if (t == AbilityType.ALTERNATE_COST || t == AbilityType.ADDITIONAL_COST
                    || t == AbilityType.COST_REDUCTION) {
                lastCostIdx = i;
            } else if (firstNonCostIdx == -1) {
                firstNonCostIdx = i;
            }
        }
        if (lastCostIdx >= 0 && firstNonCostIdx >= 0) {
            assertTrue(lastCostIdx < firstNonCostIdx,
                    "All cost lines should appear before non-cost lines: " + zeph.abilities());
        }
    }

    // --- Layout detection ---

    @Test
    void layoutDetection() {
        assertEquals("transform", convert("Name:A", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AlternateMode:DoubleFaced", "Oracle:", "ALTERNATE",
                "Name:B", "Types:Creature Werewolf", "PT:2/2", "Oracle:").layout());
        assertEquals("split", convert("Name:Fire", "ManaCost:1 R", "Types:Instant",
                "AlternateMode:Split", "Oracle:", "ALTERNATE",
                "Name:Ice", "ManaCost:1 U", "Types:Instant", "Oracle:").layout());
        assertEquals("adventure", convert("Name:Bonecrusher Giant", "ManaCost:2 R", "Types:Creature Giant",
                "PT:4/3", "AlternateMode:Adventure", "Oracle:", "ALTERNATE",
                "Name:Stomp", "ManaCost:1 R", "Types:Instant Adventure", "Oracle:").layout());
        assertEquals("meld", convert("Name:The Mightstone and Weakstone", "ManaCost:5",
                "Types:Legendary Artifact Powerstone",
                "MeldPair:Urza, Lord Protector", "AlternateMode:Meld", "Oracle:").layout());
    }

    // --- Meld halves ---

    @ParameterizedTest
    @CsvSource({
            "t/the_mightstone_and_weakstone.txt, the mightstone and weakstone",
            "p/phyrexian_dragon_engine.txt,       phyrexian dragon engine",
    })
    void meldHalfHasSingleFace(String file, String expectedName) {
        MultiCard result = convertFromFile(file);
        assertEquals("meld", result.layout());
        assertEquals(1, result.faces().size());
        assertEquals(expectedName, result.faces().get(0).name());
    }

    // --- Implicit land mana abilities ---

    @Test
    void basicLandHasImplicitManaAbility() {
        CardFace card = face("f/forest.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertEquals("{T}: add {G}", activated.get(0).descriptionText());
    }

    @Test
    void dualLandHasCombinedManaAbility() {
        CardFace card = face("b/bayou.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertTrue(activated.get(0).descriptionText().contains("{B}"));
        assertTrue(activated.get(0).descriptionText().contains("or"));
        assertTrue(activated.get(0).descriptionText().contains("{G}"));
    }

    @Test
    void nonLandDoesNotGetImplicitManaAbility() {
        CardFace card = face("l/llanowar_elves.txt");
        // Should have its own explicit activated ability, not an implicit land one
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertTrue(activated.get(0).descriptionText().contains("add {G}"));
    }

    // --- Smoke tests: cards that previously caused errors ---

    @ParameterizedTest
    @CsvSource({
            "b/bloom_tender.txt,    bloom tender",
            "f/faeburrow_elder.txt, faeburrow elder",
            "t/tarnation_vista.txt, tarnation vista",
    })
    void cardConvertsWithoutError(String file, String expectedName) {
        assertEquals(expectedName, face(file).name());
    }

    // --- Sub-ability description walking ---

    @Test
    void spellWithSubAbilityDescription() {
        CardFace card = face("a/aetherspouts.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("each attacking creature"));
    }

    @Test
    void spellWithMultipleSubAbilityDescriptions() {
        // Seed Spark has SpellDescription on both the main SP$ and on a SubAbility SVar.
        // The root SP$ becomes one SpellEffect with the sub-ability as a child.
        CardFace card = face("s/seed_spark.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        Ability root = spells.get(0);
        assertTrue(root.descriptionText().contains("destroy target artifact or enchantment"));
        assertEquals(1, root.subAbilities().size());
        assertTrue(root.subAbilities().get(0).descriptionText().contains("create two 1/1 green saproling"));
    }

    @Test
    void activatedAbilityWithSubAbilityDescription() {
        // Saprazzan Breaker's activated ability has SpellDescription on both the
        // main AB$ and a sub-ability SVar. Both descriptions are concatenated into
        // the single activated line so one oracle paragraph maps to one output line.
        CardFace card = face("s/saprazzan_breaker.txt");
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
    void activatedAbilityWithSubAbilityNoDuplicateText() {
        // Simple activated abilities (no sub-ability SpellDescription) must NOT
        // produce duplicate description text from the chain walk.
        CardFace card = face("l/llanowar_elves.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        String desc = activated.get(0).descriptionText();
        // Count occurrences of "add" — should appear exactly once
        long addCount = desc.chars().mapToObj(i -> desc.substring(Math.max(0, desc.indexOf("add"))))
                .count();
        assertFalse(desc.matches(".*add.*add.*"), "Description should not contain duplicate text: " + desc);
    }

    @Test
    void activatedAbilityWithSubAbilityDescriptionSkipped() {
        // Arachnus Spinner's activated ability has SpellDescription only on the
        // sub-ability (DBChange), not on the main AB$ Pump. The sub-ability chain
        // walk must NOT cause this to be picked up as a spell — only spells should
        // walk the chain for SpellDescription.
        CardFace card = face("a/arachnus_spinner.txt");
        assertEquals(0, countOfType(card, AbilityType.SPELL),
                "Activated ability with sub-ability description should not appear as spell");
    }

    // --- Charm sub-ability description walking ---

    @Test
    void charmWithSubAbilityDescription() {
        // "What Must Be Done" has no SpellDescription on the Charm itself,
        // so choices become top-level OPTION abilities (no parent SPELL).
        var options = allOptions(face("w/what_must_be_done.txt"));
        assertEquals(2, options.size());
        assertTrue(options.get(0).formatLine().contains("destroy all artifacts and creatures"));
        assertTrue(options.get(1).formatLine().contains("release juno"));
        assertTrue(options.get(1).formatLine().contains("return target historic permanent card"));
        assertFalse(options.get(1).formatLine().contains("Artifacts, legendaries, and Sagas are historic"),
                "Reminder text should be stripped");
    }

    @Test
    void charmChoiceWithNoDescriptionAnywhere_throws() {
        assertThrows(Exception.class, () -> convert(
                "Name:Bad Charm", "ManaCost:1 U", "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBNoop,DBDraw",
                "SVar:DBNoop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card.",
                "Oracle:"));
    }

    @Test
    void charmChoiceDescriptionOnSubAbility() {
        // Charm without SpellDescription → choices become top-level OPTION abilities
        var options = allOptions(convert(
                "Name:Chain Charm", "ManaCost:1 G", "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBChainTop,DBDirect",
                "SVar:DBChainTop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0 | SubAbility$ DBChainDesc",
                "SVar:DBChainDesc:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card from the chain.",
                "SVar:DBDirect:DB$ GainLife | LifeAmount$ 3 | SpellDescription$ Gain 3 life.",
                "Oracle:").faces().get(0));
        assertEquals(2, options.size());
        assertTrue(options.get(0).formatLine().contains("draw a card from the chain"));
        assertTrue(options.get(1).formatLine().contains("gain 3 life"));
    }

    // --- Pattern 3: Draft-only cards must not duplicate lines as text: fallback ---

    @Test
    void draftOnlyCardNoDuplicateTextFallback() {
        // Cogwork Librarian has only Draft: fields and no game-engine abilities.
        // It should emit exactly those 2 draft lines — the oracle fallback must NOT
        // also fire and add the same content again as text: lines.
        CardFace card = face("c/cogwork_librarian.txt");
        var draft = abilitiesOfType(card, AbilityType.DRAFT);
        assertEquals(2, draft.size(), "Should have exactly 2 draft lines: " + card.abilities());
        assertEquals(0, countOfType(card, AbilityType.TEXT),
                "Should have no text: fallback duplicates: " + card.abilities());
        assertEquals(2, card.abilities().size(),
                "Total abilities should be exactly 2: " + card.abilities());
    }

    // --- Adventure SA filtering ---

    @Test
    void adventureMainFaceExcludesAdventureSa() {
        // Bonecrusher Giant (main face) + Stomp (adventure face).
        // The main face should only contain Bonecrusher Giant's triggered ability,
        // NOT Stomp's spell ability ("damage can't be prevented this turn").
        MultiCard result = convertFromFile("b/bonecrusher_giant_stomp.txt");
        assertEquals("adventure", result.layout());
        CardFace main = result.faces().get(0);
        assertEquals("bonecrusher giant", main.name());
        // Main face should have the triggered ability
        assertFalse(abilitiesOfType(main, AbilityType.TRIGGERED).isEmpty(),
                "Main face should have its triggered ability: " + main.abilities());
        // Main face must NOT contain Stomp's spell
        assertTrue(abilitiesOfType(main, AbilityType.SPELL).isEmpty(),
                "Main face must not include adventure spell: " + main.abilities());
        assertFalse(main.abilities().stream()
                        .anyMatch(a -> a.descriptionText().contains("damage can't be prevented")),
                "Stomp's effect must not leak onto main face: " + main.abilities());
        // Adventure face should have the spell
        CardFace adventure = result.faces().get(1);
        assertEquals("stomp", adventure.name());
        assertFalse(abilitiesOfType(adventure, AbilityType.SPELL).isEmpty(),
                "Adventure face should have its spell: " + adventure.abilities());
    }

    // --- Draft lines ---

    @Test
    void draftLinesEmittedBeforeOtherAbilities() {
        // Aether Searcher has two Draft: lines and one triggered ability.
        // All three should appear; draft lines should precede the triggered ability.
        CardFace card = face("a/aether_searcher.txt");
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

    // --- Pattern 1: ABILITY placeholder → choose-one charm expansion ---

    @Test
    void chooseOneTriggerExpandsAbilityPlaceholder() {
        // Suncleanser: "When CARDNAME enters, ABILITY" where Execute$ is a DB$ Charm.
        // Should emit: TRIGGERED with "choose one" + 2 OPTION sub-abilities.
        CardFace card = face("s/suncleanser.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have triggered ability: " + card.abilities());
        Ability t = triggered.get(0);
        assertTrue(t.descriptionText().contains("choose one"),
                "ABILITY placeholder must be replaced with 'choose one': " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(2, options.size(), "Should have 2 charm options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION),
                "Options must be OPTION type: " + options);
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("remove all counters")),
                "Option 1 must mention counter removal: " + options);
        assertTrue(options.stream().anyMatch(o -> o.descriptionText().contains("loses all counters")),
                "Option 2 must mention opponent losing counters: " + options);
    }

    // --- Pattern A: Haunt keyword ---

    @Test
    void hauntNonCreatureEmitsSpellEffectAndTriggers() {
        // Seize the Soul is a non-creature haunt spell.
        // Expected: 1 SPELL (the primary effect) + 2 TRIGGERED (haunt keyword + haunted-dies trigger).
        CardFace card = face("s/seize_the_soul.txt");
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
        CardFace card = face("b/blind_hunter.txt");
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

    // --- Pattern B: Visit / Attraction keyword ---

    @Test
    void visitAttractionEmitsFullTriggerDescription() {
        // Storybook Ride is an Attraction with Visit keyword.
        // Expected: 1 TRIGGERED whose description starts with "visit —" and contains the exile effect.
        CardFace card = face("s/storybook_ride.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Visit attraction should emit a TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        assertTrue(visit.descriptionText().toLowerCase().contains("visit"),
                "Visit triggered description should contain 'visit': " + visit.descriptionText());
        assertTrue(visit.descriptionText().contains("exile the top"),
                "Visit triggered description should contain the exile effect: " + visit.descriptionText());
        assertFalse(visit.descriptionText().equalsIgnoreCase("visit:trigexile"),
                "Visit should not be emitted as raw 'visit:trigexile': " + visit.descriptionText());
    }

    // --- Pattern 6: Visit duplicate suppression ---

    @Test
    void visitBumperCarsNoDuplicateLines() {
        // SpellDescription starts with "Visit — " → Forge TrigDesc doubles it to "Visit — Visit — …"
        // Expected: 1 TRIGGERED with single "visit —" prefix and no SPELL sub-abilities.
        CardFace card = face("b/bumper_cars.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), "Bumper Cars should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertFalse(desc.contains("visit — visit"), "Description must not double 'visit —': " + desc);
        assertTrue(desc.contains("must be blocked"), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
    }

    @Test
    void visitFerrisWheelNoDuplicateLines() {
        CardFace card = face("f/ferris_wheel.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), "Ferris Wheel should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertFalse(desc.contains("visit — visit"), "Description must not double 'visit —': " + desc);
        assertTrue(desc.contains("phases out"), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
    }

    @Test
    void visitSwingingShipNoDuplicateLines() {
        CardFace card = face("s/swinging_ship.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), "Swinging Ship should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertFalse(desc.contains("visit — visit"), "Description must not double 'visit —': " + desc);
        assertTrue(desc.contains("additional combat phase"), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
    }

    @Test
    void visitStorybookRideNoDuplicateSpellChildren() {
        // SpellDescription doesn't start with "Visit — " → no prefix doubling, but
        // SpellEffect.fromChain produces spurious spell[1]/spell[2] sub-abilities.
        CardFace card = face("s/storybook_ride.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), "Storybook Ride should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertTrue(desc.contains("exile the top"), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
    }

    @Test
    void visitTrashBinNoDuplicateSpellChildren() {
        CardFace card = face("t/trash_bin.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size(), "Trash Bin should have exactly 1 TRIGGERED: " + card.abilities());
        Ability visit = triggered.get(0);
        String desc = visit.descriptionText().toLowerCase();
        assertTrue(desc.startsWith("visit —"), "Description should start with 'visit —': " + desc);
        assertTrue(desc.contains("mill"), "Description should contain oracle effect: " + desc);
        assertTrue(visit.subAbilities().stream().noneMatch(s -> s.type() == AbilityType.SPELL),
                "Visit triggered should have no SPELL sub-abilities: " + visit.subAbilities());
    }

    // --- Pattern C: Dice roll ResultSubAbilities ---

    @Test
    void diceRollEmitsOutcomesWithVertReplaced() {
        // Cone of Cold rolls a d20 with 3 outcome ranges.
        // Expected: 4 SPELL abilities (1 "roll a d20" + 3 outcomes with | instead of VERT).
        CardFace card = face("c/cone_of_cold.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(4, spells.size(), "Dice roll card should emit 4 SPELL (roll + 3 outcomes): " + card.abilities());
        assertTrue(spells.stream().anyMatch(a -> a.descriptionText().contains("roll a d20")),
                "Should have 'roll a d20' spell: " + spells);
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

    // --- Pattern 6: Activated dice-roll result sub-abilities as OPTION ---

    @Test
    void activatedDiceRollEmitsOutcomesAsOptions() {
        // Treasure Chest: activated RollDice with 4 result ranges.
        // Expected: 1 ACTIVATED with a single-line description and 4 OPTION sub-abilities.
        CardFace card = face("t/treasure_chest.txt");
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

    // --- Pattern 4: Tribute "if tribute wasn't paid" trigger ---

    @Test
    void tributeNotPaidTriggerIsIncluded() {
        // Pharagax Giant: K:Tribute:2 + SVar:TrigNotTribute (fires when tribute wasn't paid).
        // The keyword outputs "tribute 2"; the TrigNotTribute trigger must also be emitted
        // as a separate TRIGGERED ability — it is distinct oracle text not covered by the keyword.
        CardFace card = face("p/pharagax_giant.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(2, triggered.size(),
                "Should have 2 TRIGGERED: keyword tribute + TrigNotTribute: " + card.abilities());
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("tribute")),
                "One TRIGGERED should be the tribute keyword: " + triggered);
        assertTrue(triggered.stream().anyMatch(a -> a.descriptionText().contains("tribute wasn't paid")),
                "One TRIGGERED should be the 'if tribute wasn't paid' effect: " + triggered);
    }

    // --- Pattern 4: RepeatSubAbility chain walking ---

    @Test
    void repeatSubAbilityDescriptionIsIncluded() {
        // March of Souls: main SA destroys all creatures, then a RepeatEach SVar per player
        // creates spirit tokens. The spirit-token text lives in a RepeatSubAbility SVar and
        // must be concatenated into the single spell line via collectChainText (formatBlock).
        CardFace card = face("m/march_of_souls.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly one SPELL: " + card.abilities());
        // Use formatText (which calls formatBlock) to get the fully concatenated chain text.
        String formatted = card.formatText();
        assertTrue(formatted.contains("destroy all creatures"),
                "Output should contain 'destroy all creatures': " + formatted);
        assertTrue(formatted.contains("spirit"),
                "Output should contain spirit-token text from RepeatSubAbility: " + formatted);
    }

    // --- Pattern 8a: Activated ability with description on SubAbility$ ---

    @Test
    void activatedAbilityWithDescriptionOnSubAbility() {
        // Arachnus Spinner: activated ability has no SpellDescription on the root SA;
        // the description lives on SubAbility$ DBChange.
        CardFace card = face("a/arachnus_spinner.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(),
                "Arachnus Spinner should have 1 ACTIVATED ability: " + card.abilities());
        String formatted = card.formatText();
        assertTrue(formatted.toLowerCase().contains("search"),
                "Activated ability text should include 'search': " + formatted);
    }

    // --- Pattern 8b: Visit trigger with Charm overriding ability ---

    @Test
    void visitCharmExpandsToOptionSubAbilities() {
        // Balloon Stand: Visit trigger whose overriding ability is a Charm SA.
        // Expected: 1 TRIGGERED with "visit" in header and 2 OPTION sub-abilities,
        // not embedded bullet text inside the triggered description.
        CardFace card = face("b/balloon_stand.txt");
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

    // --- Pattern 9a: Tiered charm — PrecostDesc prepended to option text ---

    @Test
    void tieredCharmIncludesPrecostDescInOptionText() {
        // Vincent's Limit Break: Tiered charm with PrecostDesc$ on each choice SVar.
        // Option text must include the PrecostDesc name, not just the raw P/T string.
        CardFace card = face("v/vincents_limit_break.txt");
        String formatted = card.formatText();
        assertTrue(formatted.toLowerCase().contains("galian beast"),
                "Tiered charm option should include 'Galian Beast': " + formatted);
        assertTrue(formatted.toLowerCase().contains("death gigas"),
                "Tiered charm option should include 'Death Gigas': " + formatted);
        assertTrue(formatted.toLowerCase().contains("hellmasker"),
                "Tiered charm option should include 'Hellmasker': " + formatted);
    }

    // --- Pattern 9b: MayEffectFromOpeningHand keyword ---

    @Test
    void mayEffectFromOpeningHandEmitsTriggeredAbility() {
        // Chancellor of the Tangle: MayEffectFromOpeningHand:ManaOnMain where ManaOnMain
        // has no SpellDescription; description is on the RevealCard SVar fallback.
        CardFace card = face("c/chancellor_of_the_tangle.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertTrue(triggered.stream().anyMatch(a ->
                        a.descriptionText().toLowerCase().contains("may reveal")),
                "Should have a TRIGGERED ability containing 'may reveal': " + triggered);
    }

    // --- Pattern J: Sub-ability text concatenated into activated line ---

    @Test
    void activatedAbilitySubChainConcatenatedNotSplit() {
        // fleshformer: oracle=1 line but converter previously emitted 2 (root + sub-ability).
        // After fix, sub-ability text must be concatenated into the root line.
        CardFace card = face("f/fleshformer.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(), "fleshformer should have exactly 1 ACTIVATED: " + card.abilities());
        // No SPELL sub-abilities should exist (concatenated into root)
        long spellSubs = activated.get(0).subAbilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).count();
        assertEquals(0, spellSubs,
                "No SPELL sub-abilities should remain after concatenation: " + activated.get(0).subAbilities());
    }

    @Test
    void grinningSorcAbilityIsSingleLine() {
        // grinning_totem: oracle=1 line; converter must not split into 2 spell lines.
        CardFace card = face("g/grinning_totem.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size(), "grinning_totem should have exactly 1 ACTIVATED: " + card.abilities());
        long spellSubs = activated.get(0).subAbilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).count();
        assertEquals(0, spellSubs,
                "No SPELL sub-abilities after concatenation: " + activated.get(0).subAbilities());
    }

    // --- Pattern A: Triggered dice-roll ResultSubAbilities as OPTION ---

    @Test
    void triggeredDiceRollEmitsOutcomesAsOptions() {
        // Swarming Goblins: triggered RollDice with 3 result ranges.
        // Expected: 1 TRIGGERED with roll-a-d20 header and 3 OPTION sub-abilities.
        CardFace card = face("s/swarming_goblins.txt");
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
        assertTrue(options.size() >= 2,
                "Should have at least 2 OPTION sub-abilities for dice results: " + options);
    }

    // --- Pattern B: ETBReplacement / ImmediateTrigger no duplicates ---

    @Test
    void etbReplacementNoDuplicateSpellLine() {
        // Sigarda's Splendor: has an ETBReplacement/ImmediateTrigger SVar that is
        // processed both by the triggers/replacements loop and (incorrectly) by the
        // spell loop.  After the fix, no ability text should appear twice.
        CardFace card = face("s/sigardas_splendor.txt");
        String formatted = card.formatText();
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

    // --- Pattern 7: CharmNum-based "choose N" header synthesis ---

    @Test
    void chooseTwoCharmEmitsChooseTwoHeader() {
        // Atarka's Command: CharmNum$ 2 (no MinCharmNum) — exactly two modes.
        // Expected: 1 SPELL with "choose two" in description + 4 OPTION sub-abilities.
        CardFace card = face("a/atarkas_command.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        Ability spell = spells.get(0);
        assertTrue(spell.descriptionText().contains("choose two"),
                "Spell description should contain 'choose two': " + spell.descriptionText());
        List<Ability> options = spell.subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(4, options.size(), "Should have 4 OPTION sub-abilities: " + spell.subAbilities());
    }

    @Test
    void chooseOneOrBothCharmEmitsCorrectHeader() {
        // Against All Odds: MinCharmNum$ 1, CharmNum$ 2 — "choose one or both".
        CardFace card = face("a/against_all_odds.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        assertTrue(spells.get(0).descriptionText().contains("choose one or both"),
                "Spell description should contain 'choose one or both': " + spells.get(0).descriptionText());
    }

    @Test
    void chooseUpToOneCharmEmitsCorrectHeader() {
        // Ertai Resurrected trigger: MinCharmNum$ 0, no CharmNum — "choose up to one".
        // The charm is in a triggered sub-ability; check the full formatted text.
        CardFace card = face("e/ertai_resurrected.txt");
        String formatted = card.formatText();
        assertTrue(formatted.contains("choose up to one"),
                "Output should contain 'choose up to one': " + formatted);
    }

    // --- Pattern F: Saga chapter content resolved from execute SVar ---

    @Test
    void sagaChapterWithHeaderOnlyTriggerDescriptionResolvesContent() {
        // ballad_of_the_black_flag: TriggerDescription is just "I, II, III —" with
        // the effect on the execute SVar.  Chapter lines must include the effect text.
        CardFace card = face("b/ballad_of_the_black_flag.txt");
        var chapters = abilitiesOfType(card, AbilityType.CHAPTER);
        assertFalse(chapters.isEmpty(), "Should have CHAPTER abilities: " + card.abilities());
        // At least one chapter must have non-empty effect text after the roman numeral header
        assertTrue(chapters.stream().anyMatch(c -> {
            String desc = c.descriptionText();
            int dash = desc.indexOf('\u2014');
            return dash >= 0 && !desc.substring(dash + 1).trim().isEmpty();
        }), "At least one chapter should have effect text after em-dash: " + chapters);
    }

    // --- Pattern E: Ward keyword classified as STATIC ---

    @Test
    void wardKeywordClassifiedAsStatic() {
        // calim_djinn_emperor: K:Ward:2 should emit "static: ward {2}", not "triggered: ward {2}".
        CardFace card = face("c/calim_djinn_emperor.txt");
        var statics = abilitiesOfType(card, AbilityType.STATIC);
        assertTrue(statics.stream().anyMatch(a -> a.descriptionText().toLowerCase().contains("ward")),
                "Ward should be classified as STATIC: " + card.abilities());
        assertEquals(0, abilitiesOfType(card, AbilityType.TRIGGERED).stream()
                        .filter(a -> a.descriptionText().toLowerCase().contains("ward")).count(),
                "Ward must NOT appear as TRIGGERED: " + card.abilities());
    }

    // --- Pattern I: Missing spell effect after additional cost ---

    @Test
    void sheoldredRestorationEmitsSpellEffect() {
        // TriggerDescription$ on root spell SA must be used as the spell description
        var spells = abilitiesOfType(face("s/sheoldreds_restoration.txt"), AbilityType.SPELL);
        assertTrue(spells.stream().anyMatch(
                a -> a.descriptionText().toLowerCase().contains("return target creature card")),
                "sheoldred's restoration spell effect missing: " + spells);
    }

    @Test
    void vincentLimitBreakEmitsAdditionalDescription() {
        // AdditionalDescription$ on Charm SA must become the main spell line
        var spells = abilitiesOfType(face("v/vincents_limit_break.txt"), AbilityType.SPELL);
        assertTrue(spells.stream().anyMatch(
                a -> a.descriptionText().toLowerCase().contains("until end of turn")),
                "vincent's limit break missing AdditionalDescription: " + spells);
    }

    @Test
    void vincentLimitBreakOptionsIncludeModeCost() {
        // Tiered charm options must include the ModeCost in "Name — {cost} — P/T" format
        var spells = abilitiesOfType(face("v/vincents_limit_break.txt"), AbilityType.SPELL);
        assertFalse(spells.isEmpty(), "vincent's limit break should have a SPELL: " + spells);
        var options = spells.get(0).subAbilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertTrue(options.stream().anyMatch(
                a -> a.descriptionText().contains("{0}") || a.descriptionText().contains("{1}")),
                "vincent's limit break options missing ModeCost: " + options);
    }

    // --- Pattern 1 extension: ABILITY placeholder via nested structures ---

    @Test
    void immediatelyTriggeredChooseTwoExpandsAbilityPlaceholder() {
        // Caesar: ImmediateTrigger → Charm(CharmNum=2), TriggerDescription ends with "ABILITY"
        CardFace card = face("c/caesar_legions_emperor.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.get(0);
        assertFalse(t.descriptionText().contains("ABILITY"), "ABILITY must be replaced: " + t.descriptionText());
        assertTrue(t.descriptionText().contains("choose two"), "Must say 'choose two': " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(3, options.size(), "Must have 3 options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void deeplyNestedImmediatelyTriggeredChooseOneExpandsAbilityPlaceholder() {
        // Cemetery Desecrator: ChangeZone → SubAbility → ImmediateTrigger → Charm, TriggerDescription ends with "ABILITY"
        CardFace card = face("c/cemetery_desecrator.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.get(0);
        assertFalse(t.descriptionText().contains("ABILITY"), "ABILITY must be replaced: " + t.descriptionText());
        assertTrue(t.descriptionText().contains("choose one"), "Must say 'choose one': " + t.descriptionText());
        List<Ability> options = t.subAbilities();
        assertEquals(2, options.size(), "Must have 2 options: " + options);
        assertTrue(options.stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void immediatelyTriggeredChooseOneExpandsAbilityPlaceholder() {
        // Hylda: ImmediateTrigger → Charm (default choose one), 3 choices
        CardFace card = face("h/hylda_of_the_icy_crown.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        Ability t = triggered.stream()
                .filter(a -> a.descriptionText().contains("tap an untapped creature"))
                .findFirst().orElseThrow();
        assertFalse(t.descriptionText().contains("ABILITY"), "ABILITY must be replaced: " + t.descriptionText());
        assertTrue(t.descriptionText().contains("choose one"), "Must say 'choose one': " + t.descriptionText());
        assertEquals(3, t.subAbilities().size(), "Must have 3 options: " + t.subAbilities());
    }

    @Test
    void hauntCharmExpandsAbilityPlaceholder() {
        // Orzhov Pontiff: Haunt SVar is DB$ Charm, no SpellDescription → ABILITY stays unexpanded today
        CardFace card = face("o/orzhov_pontiff.txt");
        // The haunted-dies triggered ability should mention "choose one", not "ABILITY"
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        Ability hauntTrig = triggered.stream()
                .filter(a -> a.descriptionText().contains("haunts dies"))
                .findFirst().orElseThrow();
        assertFalse(hauntTrig.descriptionText().contains("ABILITY"),
                "ABILITY must be replaced: " + hauntTrig.descriptionText());
        assertTrue(hauntTrig.descriptionText().contains("choose one"),
                "Must say 'choose one': " + hauntTrig.descriptionText());
        assertEquals(2, hauntTrig.subAbilities().size(), "Must have 2 options: " + hauntTrig.subAbilities());
        assertTrue(hauntTrig.subAbilities().stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void chapterCharmExpandsAbilityPlaceholder() {
        // Life of Toshiro: Chapter I/II execute is DB$ Charm, SpellDescription$ ABILITY
        MultiCard card = convertFromFile("l/life_of_toshiro_umezawa_memory_of_toshiro.txt");
        CardFace front = card.faces().get(0);
        var chapters = abilitiesOfType(front, AbilityType.CHAPTER);
        // Find chapter I/II (the charm chapter)
        Ability charmChapter = chapters.stream()
                .filter(c -> c.descriptionText().contains("I") && c.descriptionText().contains("II"))
                .findFirst().orElseThrow();
        assertFalse(charmChapter.descriptionText().contains("ABILITY"),
                "ABILITY must be replaced: " + charmChapter.descriptionText());
        assertTrue(charmChapter.descriptionText().contains("choose one"),
                "Must say 'choose one': " + charmChapter.descriptionText());
        assertEquals(3, charmChapter.subAbilities().size(),
                "Must have 3 charm options: " + charmChapter.subAbilities());
        assertTrue(charmChapter.subAbilities().stream().allMatch(o -> o.type() == AbilityType.OPTION));
    }

    @Test
    void diceRollTriggerExpandsAbilityPlaceholder() {
        // Delina: Trigger → Repeat → RollDice, TriggerDescription ends with "ABILITY"
        CardFace card = face("d/delina_wild_mage.txt");
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

    // --- Pattern 2: cost-reduction clause on keyword activated abilities ---

    @Test
    void equipReduceCostAppendsText() {
        // Plate Armor: K:Equip:3:::ReduceCost$ Y:This ability costs {1} less…
        CardFace card = face("p/plate_armor.txt");
        Ability equip = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("equip"))
                .findFirst().orElseThrow();
        assertTrue(equip.descriptionText().contains("costs {1} less"),
                "Must include cost reduction: " + equip.descriptionText());
        assertTrue(equip.descriptionText().contains("each other equipment"),
                "Must include count clause: " + equip.descriptionText());
    }

    @Test
    void equipVariableReduceCostAppendsText() {
        // Belt of Giant Strength: ReduceCost$ X:This ability costs {X} less…
        CardFace card = face("b/belt_of_giant_strength.txt");
        Ability equip = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("equip"))
                .findFirst().orElseThrow();
        assertTrue(equip.descriptionText().contains("costs {X} less"),
                "Must include variable cost reduction: " + equip.descriptionText());
        assertTrue(equip.descriptionText().contains("power of the creature"),
                "Must include power clause: " + equip.descriptionText());
    }

    @Test
    void adaptReduceCostAppendsText() {
        // Pteramander: K:Adapt:4:7 U:X:instant and sorcery card in your graveyard
        CardFace card = face("p/pteramander.txt");
        Ability adapt = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("adapt"))
                .findFirst().orElseThrow();
        assertTrue(adapt.descriptionText().contains("costs {1} less"),
                "Must include cost reduction: " + adapt.descriptionText());
        assertTrue(adapt.descriptionText().contains("instant and sorcery card in your graveyard"),
                "Must include count clause: " + adapt.descriptionText());
    }

    @Test
    void adaptArtifactReduceCostAppendsText() {
        // Etherium Pteramander: K:Adapt:4:6 B:X:other artifact you control
        CardFace card = face("e/etherium_pteramander.txt");
        Ability adapt = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("adapt"))
                .findFirst().orElseThrow();
        assertTrue(adapt.descriptionText().contains("costs {1} less"),
                "Must include cost reduction: " + adapt.descriptionText());
        assertTrue(adapt.descriptionText().contains("other artifact you control"),
                "Must include count clause: " + adapt.descriptionText());
    }

    @Test
    void monstrosityReduceCostAppendsText() {
        // Grim Giganotosaurus: K:Monstrosity:10:10 B G:X:creature with power 4 or greater…
        CardFace card = face("g/grim_giganotosaurus.txt");
        Ability monstrosity = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("monstrosity"))
                .findFirst().orElseThrow();
        assertTrue(monstrosity.descriptionText().contains("costs {1} less"),
                "Must include cost reduction: " + monstrosity.descriptionText());
        assertTrue(monstrosity.descriptionText().contains("creature with power 4"),
                "Must include count clause: " + monstrosity.descriptionText());
    }

    @Test
    void specializeReduceCostAppendsText() {
        // Imoen, Trickster Friend: K:Specialize:5::This ability costs {3} less…:ReduceCost$ X
        CardFace card = face("i/imoen_trickster_friend.txt");
        Ability specialize = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("specialize"))
                .findFirst().orElseThrow();
        assertTrue(specialize.descriptionText().contains("costs {3} less"),
                "Must include cost reduction: " + specialize.descriptionText());
        assertTrue(specialize.descriptionText().contains("instant and/or sorcery cards in your graveyard"),
                "Must include count clause: " + specialize.descriptionText());
    }

    // --- Pattern 3: Craft ability terse format ---

    @Test
    void craftWithIslandTypeAndCost() {
        // Waterlogged Hulk: K:Craft:3 U ExileCtrlOrGrave<1/Island.Other>
        CardFace card = face("w/waterlogged_hulk_watertight_gondola.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with island"),
                "Must include type 'island': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{3}{U}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithArtifactTypeAndCost() {
        // Oteclan Landmark: K:Craft:2 W ExileCtrlOrGrave<1/Artifact.Other>
        CardFace card = face("o/oteclan_landmark_oteclan_levitator.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with artifact"),
                "Must include type 'artifact': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{2}{W}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithCreatureTypeAndCost() {
        // Tithing Blade: K:Craft:4 B ExileCtrlOrGrave<1/Creature.Other>
        CardFace card = face("t/tithing_blade_consuming_sepulcher.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with creature"),
                "Must include type 'creature': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{4}{B}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithArtifactMulticolorCost() {
        // Dire Flail: K:Craft:3 R R ExileCtrlOrGrave<1/Artifact.Other>
        CardFace card = face("d/dire_flail_dire_blunderbuss.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with artifact"),
                "Must include type 'artifact': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{3}{R}{R}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithVariableCountOneOrMore() {
        // Sunbird Standard: K:Craft:5 XMin1 ExileCtrlOrGrave<X/Permanent.Other/permanent>
        CardFace card = face("s/sunbird_standard_sunbird_effigy.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with one or more"),
                "Must include 'one or more': " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{5}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithExplicitMultiTypeDescription() {
        // Throne of the Grim Captain: K:Craft:4 ExileCtrlOrGrave<…>:a Dinosaur, a Merfolk, a Pirate, and a Vampire:the four
        CardFace card = face("t/throne_of_the_grim_captain_the_grim_captain.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with a dinosaur"),
                "Must include explicit type description: " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{4}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    @Test
    void craftWithTypeShareConstraintDescription() {
        // Eye of Ojer Taq: K:Craft:6 ExileCtrlOrGrave<2/…>:two that share a card type:the two
        CardFace card = face("e/eye_of_ojer_taq_apex_observatory.txt");
        Ability craft = abilitiesOfType(card, AbilityType.ACTIVATED).stream()
                .filter(a -> a.descriptionText().contains("craft"))
                .findFirst().orElseThrow();
        assertTrue(craft.descriptionText().contains("craft with two that share a card type"),
                "Must include constraint description: " + craft.descriptionText());
        assertTrue(craft.descriptionText().contains("{6}"),
                "Must include mana cost: " + craft.descriptionText());
        assertFalse(craft.descriptionText().contains("exile"),
                "Must not include verbose exile clause: " + craft.descriptionText());
    }

    // --- Pattern 4: Missing secondary effects ---

    // Regression: delayed-trigger execute SVars must not bleed into the trigger description.
    // DelTrigBlocked/DelTrigBlocker have TriggerDescription$ which is already summarised
    // in the root TriggerDescription; concatenating it would duplicate the text.
    @Test
    void delayedTriggerExecuteNotConcatenatedIntoTrigger() {
        CardFace card = face("s/sawtooth_ogre.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have TRIGGERED: " + card.abilities());
        for (Ability t : triggered) {
            String desc = t.descriptionText();
            // "1 damage" must appear exactly once — duplication would cause two occurrences.
            int first = desc.indexOf("1 damage");
            assertTrue(first >= 0, "Must contain '1 damage': " + desc);
            assertEquals(-1, desc.indexOf("1 damage", first + 1),
                    "Damage clause must not be duplicated: " + desc);
        }
    }

    @Test
    void delayedTriggerDestroyNotConcatenatedIntoTrigger() {
        CardFace card = face("t/tangle_asp.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty(), "Should have TRIGGERED: " + card.abilities());
        for (Ability t : triggered) {
            String desc = t.descriptionText();
            // "destroy" must appear exactly once — duplication would cause two occurrences.
            int first = desc.indexOf("destroy");
            assertTrue(first >= 0, "Must contain 'destroy': " + desc);
            assertEquals(-1, desc.indexOf("destroy", first + 1),
                    "Destroy clause must not be duplicated: " + desc);
        }
    }

    // Regression: StackDescription on a sub-ability that copies parent SpellDescription verbatim
    // must not produce a duplicate spell line (e.g. Bifurcate DBChangeZone StackDescription).
    @Test
    void stackDescriptionDuplicatingParentNotEmitted() {
        CardFace card = face("b/bifurcate.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        String desc = spells.get(0).descriptionText();
        // "search" must appear exactly once (StackDescription duplication would cause two)
        int first = desc.indexOf("search");
        assertTrue(first >= 0, "Must contain 'search': " + desc);
        assertEquals(-1, desc.indexOf("search", first + 1),
                "search clause must not be duplicated: " + desc);
    }

    // Regression: replacement-effect Description$ that exactly matches parent SpellDescription
    // must not be emitted as a sub-ability (e.g. Shadowbane RepDmg.Description$).
    @Test
    void replacementDescriptionDuplicatingParentNotEmitted() {
        CardFace card = face("s/shadowbane.txt");
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
        CardFace card = face("e/energy_arc.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        String desc = spells.get(0).descriptionText();
        // "prevent" must appear exactly once
        int first = desc.indexOf("prevent");
        assertTrue(first >= 0, "Must contain 'prevent': " + desc);
        assertEquals(-1, desc.indexOf("prevent", first + 1),
                "prevent clause must not be duplicated: " + desc);
    }

    // Regression: replacement-effect Description$ containing Forge placeholder EFFECTSOURCE
    // must not be emitted as oracle text (e.g. Delirium RPrevent1).
    @Test
    void replacementDescriptionWithEffectsourceNotEmitted() {
        CardFace card = face("d/delirium.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have exactly 1 SPELL: " + card.abilities());
        String formatted = card.formatText();
        // "prevent" must appear exactly once (from SpellDescription, not from replacement SVar)
        int first = formatted.indexOf("prevent");
        assertTrue(first >= 0, "Must contain 'prevent': " + formatted);
        assertEquals(-1, formatted.indexOf("prevent", first + 1),
                "prevent clause from EFFECTSOURCE replacement must not duplicate: " + formatted);
    }

    // Regression: a trigger SVar referenced in DB$ Effect.Triggers$ that has TriggerZones$
    // must not be emitted as a spell sub-ability (it's also a top-level T: trigger).
    @Test
    void triggerZonesSVarNotEmittedAsSpellSubAbility() {
        CardFace card = face("e/ertais_meddling.txt");
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
        CardFace card = face("b/bounty_of_might.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size(), "Should have 1 SPELL: " + card.abilities());
        // The single SPELL ability has a nested sub-ability chain representing all 3 effects.
        // collectChainText must produce all 3 repetitions (not deduplicated away).
        String formatted = card.formatText();
        // Count occurrences of "+3/+3" — must appear 3 times
        int count = 0, idx = 0;
        while ((idx = formatted.indexOf("+3/+3", idx)) >= 0) { count++; idx++; }
        assertEquals(3, count, "All 3 identical pump effects must be present: " + formatted);
    }

    @Test
    void triggerExecuteChainTrailingTextConcatenated() {
        // Oil-Gorger Troll: execute chain (TrigGainLife → DBDraw) has SpellDescription
        // that must be concatenated into the trigger description.
        CardFace card = face("o/oil_gorger_troll.txt");
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
        CardFace card = face("m/mage_hunters_onslaught.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty(), "Should have SPELL: " + card.abilities());
        String formatted = card.formatText();
        assertTrue(formatted.contains("destroy target creature or planeswalker"),
                "Must contain first effect: " + formatted);
        assertTrue(formatted.contains("whenever a creature blocks"),
                "Must contain trigger description from Effect SVar: " + formatted);
    }

    @Test
    void effectTriggerDescriptionConcatenatedIntoSpellLine() {
        // Reckless Blaze: DB$ Effect with Triggers$ DiesTrig.
        // TriggerDescription must appear in the output (oracle=1 single paragraph).
        CardFace card = face("r/reckless_blaze.txt");
        String formatted = card.formatText();
        assertTrue(formatted.contains("deals 5 damage to each creature"),
                "Must contain main effect: " + formatted);
        assertTrue(formatted.contains("dies this turn"),
                "Must contain trigger description from Effect SVar: " + formatted);
    }

    @Test
    void effectReplacementDescriptionConcatenated() {
        // Dazzling Reflection: DB$ Effect with ReplacementEffects$ Dazzle.
        // Dazzle's Description$ must appear in the output.
        CardFace card = face("d/dazzling_reflection.txt");
        String formatted = card.formatText();
        assertTrue(formatted.contains("you gain life equal to target creature"),
                "Must contain main effect: " + formatted);
        assertTrue(formatted.contains("prevent that damage"),
                "Must contain replacement Description$ from Effect SVar: " + formatted);
    }

    @Test
    void stackDescriptionFallbackForSubAbility() {
        // Transgress the Mind: ExileCard SVar has StackDescription but no SpellDescription.
        // Must concatenate the StackDescription text into the spell line.
        CardFace card = face("t/transgress_the_mind.txt");
        String formatted = card.formatText();
        assertTrue(formatted.contains("target player reveals their hand"),
                "Must contain root spell description: " + formatted);
        assertTrue(formatted.contains("mana value 3 or greater"),
                "Must contain sub-ability StackDescription text: " + formatted);
    }

    // Regression: replacement-effect Description$ containing EFFECTSOURCE in an activated ability
    // must not be emitted even when ActivatedAbilityEntry calls fromChain() without parentDesc.
    @Test
    void replacementDescriptionWithEffectsourceNotEmittedFromActivatedAbility() {
        // Stuffy Doll Avatar: SelflessDamage.Description$ contains EFFECTSOURCE placeholder.
        // ActivatedAbilityEntry calls fromChain(sub) without parentDesc so parentDesc=null;
        // the effectsource guard must fire before the parentDesc null check.
        CardFace card = face("s/stuffy_doll_avatar.txt");
        String formatted = card.formatText();
        assertFalse(formatted.contains("effectsource"),
                "EFFECTSOURCE placeholder must not appear in output: " + formatted);
    }

    // Regression: DB$ Effect SA whose SpellDescription is entirely reminder text (strips to empty)
    // must not trigger collectEffectDescriptions — its replacement children are already covered.
    @Test
    void replacementDescriptionOfAllReminderTextEffectNotEmitted() {
        // Sarah's Wings: NoDamage SVar has SpellDescription$ = "(Players with flying can't...)"
        // which is entirely reminder text. collectEffectDescriptions must not run on it.
        CardFace card = face("s/sarahs_wings.txt");
        String formatted = card.formatText();
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
        CardFace card = face("e/eye_for_an_eye.txt");
        String formatted = card.formatText();
        long count = formatted.lines()
                .filter(l -> l.contains("deals that much damage to that source"))
                .count();
        assertEquals(1, count,
                "The deals-damage effect must appear exactly once (not doubled): " + formatted);
    }

    @Test
    void multiProtectionKeywordsEachEmitSeparateStaticLine() {
        // Elite Inquisitor has K:Protection:Vampire / K:Protection:Werewolf / K:Protection:Zombie.
        // Oracle groups them as "Protection from Vampires, from Werewolves, and from Zombies"
        // (one line), but the converter correctly models each Forge keyword as its own static
        // ability.  Pin this behaviour so it is not accidentally collapsed.
        var statics = abilitiesOfType(face("e/elite_inquisitor.txt"), AbilityType.STATIC);
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
        var statics = abilitiesOfType(face("j/jaheira_harper_emissary.txt"), AbilityType.STATIC);
        long hexproofs = statics.stream()
                .filter(a -> a.descriptionText().toLowerCase().contains("hexproof from"))
                .count();
        assertEquals(2, hexproofs,
                "Each of the two hexproof keywords must produce its own static line: " + statics);
    }

    // --- Pattern 8: Missing spell description / Pattern 9: NICKNAME / Pattern 10: Developer note ---

    @Test
    void crashingWaveAdditionalCostPresentNoSpellEffect() {
        // A:SP$ Tap has no SpellDescription/StackDescription → no spell effect can be emitted.
        var card = face("c/crashing_wave.txt");
        assertEquals(1, countOfType(card, AbilityType.ADDITIONAL_COST));
        assertTrue(abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0)
                   .descriptionText().contains("waterbend"));
        assertEquals(0, countOfType(card, AbilityType.SPELL),
                     "No SpellDescription in Forge script → no spell effect emitted: " + card.abilities());
    }

    @Test
    void silvosActivatedAbilityUsesCardnamePlaceholder() {
        // SpellDescription$ Regenerate CARDNAME. — Java side already correct.
        var card = face("s/silvos_rogue_elemental.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertTrue(activated.get(0).descriptionText().contains("regenerate"));
        assertFalse(activated.get(0).descriptionText().toLowerCase().contains("silvos"),
                    "Card name must be replaced, not kept literal: " + activated);
    }

    @Test
    void nicknameReplacedWithCardname() {
        // NICKNAME is a Forge alias for the card name — must appear as CARDNAME in output.
        CardFace card = convert(
            "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
            "A:AB$ Pump | Cost$ 1 | SpellDescription$ NICKNAME gets +2/+2 until end of turn.",
            "Oracle:Test Card gets +2/+2 until end of turn."
        ).faces().get(0);
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertTrue(activated.get(0).descriptionText().contains("CARDNAME"),
                   "NICKNAME must be replaced with CARDNAME: " + activated);
        assertFalse(activated.get(0).descriptionText().contains("NICKNAME"),
                    "NICKNAME must not remain in output: " + activated);
    }

    @Test
    void developerNoteStrippedFromTextField() {
        // Text:[Developer's note: …] is not oracle content; must not be emitted.
        var card = face("c/celestine_cave_witch.txt");
        assertNull(card.text(), "Developer's note should produce null text field, got: " + card.text());
    }

    @Test
    void goblinPolkaBandDeveloperNoteStripped() {
        var card = face("g/goblin_polka_band.txt");
        assertNull(card.text(), "Developer's note should produce null text field, got: " + card.text());
    }

    // --- Helpers ---

    private void assertCostsBeforeSpells(CardFace card) {
        var abilities = card.abilities();
        int lastCostIdx = -1, firstSpellIdx = -1;
        for (int i = 0; i < abilities.size(); i++) {
            AbilityType t = abilities.get(i).type();
            if (t == AbilityType.ADDITIONAL_COST || t == AbilityType.ALTERNATE_COST
                    || t == AbilityType.COST_REDUCTION) {
                lastCostIdx = i;
            } else if (firstSpellIdx == -1 && t == AbilityType.SPELL) {
                firstSpellIdx = i;
            }
        }
        if (lastCostIdx >= 0 && firstSpellIdx >= 0) {
            assertTrue(lastCostIdx < firstSpellIdx,
                    "Costs should appear before spells: " + abilities);
        }
    }

    private void assertNoRawEtbReplacementKeyword(CardFace card) {
        long raw = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.STATIC || a.type() == AbilityType.ACTIVATED)
                        && a.descriptionText().contains("etbreplacement"))
                .count();
        assertEquals(0, raw, "ETBReplacement should not appear as raw keyword");
    }
}
