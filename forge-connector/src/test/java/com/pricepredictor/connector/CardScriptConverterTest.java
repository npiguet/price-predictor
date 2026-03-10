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
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
@ExtendWith(ForgeExtension.class)
class CardScriptConverterTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private final CardScriptConverter converter = new CardScriptConverter();

    private MultiCard convert(String... lines) {
        return converter.convertCard(Arrays.asList(lines), "test.txt");
    }

    private MultiCard convertFromFile(String relativePath) {
        try {
            Path file = CARDS_FOLDER.resolve(relativePath);
            List<String> lines = Files.readAllLines(file);
            return converter.convertCard(lines, file.getFileName().toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private ConvertedCard face(String relativePath) {
        return convertFromFile(relativePath).faces().get(0);
    }

    private List<AbilityLine> abilitiesOfType(ConvertedCard card, AbilityType type) {
        return card.abilities().stream().filter(a -> a.type() == type).toList();
    }

    private long countOfType(ConvertedCard card, AbilityType type) {
        return card.abilities().stream().filter(a -> a.type() == type).count();
    }

    private void assertHasAbility(ConvertedCard card, AbilityType type, String... containsTexts) {
        var abilities = abilitiesOfType(card, type);
        assertFalse(abilities.isEmpty(), "Expected at least one " + type + " but got: " + card.abilities());
        for (String text : containsTexts) {
            assertTrue(abilities.stream().anyMatch(a -> a.description().contains(text)),
                    "Expected '" + text + "' in " + type + " abilities but got: " + abilities);
        }
    }

    // --- US1: Basic card conversion ---

    @Test
    void vanillaCreature() {
        ConvertedCard card = face("g/grizzly_bears.txt");
        assertEquals("grizzly bears", card.name());
        assertTrue(card.manaCost().contains("{G}"));
        assertEquals("creature bear", card.types());
        assertEquals("2/2", card.powerToughness());
        assertTrue(card.abilities().isEmpty());
        assertNull(card.loyalty());
    }

    @Test
    void passiveKeywords() {
        var keywords = abilitiesOfType(face("s/serra_angel.txt"), AbilityType.KEYWORD_PASSIVE);
        assertEquals(2, keywords.size());
        assertEquals("keyword: flying", keywords.get(0).formatLine());
        assertEquals("keyword: vigilance", keywords.get(1).formatLine());
        assertNull(keywords.get(0).actionNumber());
    }

    @Test
    void activatedAbility() {
        String line = abilitiesOfType(face("l/llanowar_elves.txt"), AbilityType.ACTIVATED).get(0).formatLine();
        assertTrue(line.startsWith("activated[1]:"), line);
        assertTrue(line.contains("{T}"), line);
        assertTrue(line.contains("add {G}"), line);
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
    void engineMetadataExcluded() {
        String output = OutputFormatter.formatMultiCard(convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AI:RemoveDeck:Random", "DeckHints:Ability$LifeGain",
                "SVar:AnotherSVar:DB$ Something", "Oracle:"));
        assertFalse(output.contains("AI:"));
        assertFalse(output.contains("DeckHints"));
        assertFalse(output.contains("SVar"));
    }

    @Test
    void actionCounterIncrementsAcrossMixedTypes() {
        ConvertedCard card = convert(
                "Name:Versatile Card", "ManaCost:2 W", "Types:Creature Human", "PT:2/2",
                "K:Cycling:2", "A:AB$ GainLife | Cost$ T | LifeAmount$ 1 | SpellDescription$ Gain 1 life.",
                "Oracle:").faces().get(0);
        List<AbilityLine> actionable = card.abilities().stream()
                .filter(a -> a.actionNumber() != null).toList();
        assertTrue(actionable.size() >= 2);
        for (int i = 0; i < actionable.size() - 1; i++) {
            assertTrue(actionable.get(i).actionNumber() < actionable.get(i + 1).actionNumber());
        }
    }

    @Test
    void textCasingCorrect() {
        String output = OutputFormatter.formatMultiCard(convertFromFile("l/llanowar_elves.txt"));
        assertTrue(output.contains("llanowar elves"));
        assertTrue(output.contains("{G}"));
        assertTrue(output.contains("{T}"));
    }

    @Test
    void variableXStaysUppercase() {
        // Standalone X preserved
        assertEquals("-X: deal X damage", CardScriptConverter.applyTextCasing("-X: Deal X damage"));
        assertEquals("+X/+0 until end of turn", CardScriptConverter.applyTextCasing("+X/+0 until end of turn"));
        assertEquals("where X is the number", CardScriptConverter.applyTextCasing("Where X is the number"));
        // X inside words lowercased
        assertEquals("exile target creature", CardScriptConverter.applyTextCasing("Exile target creature"));
        assertEquals("next end step", CardScriptConverter.applyTextCasing("Next end step"));
        assertEquals("tax each opponent", CardScriptConverter.applyTextCasing("Tax each opponent"));
        // Braces
        assertEquals("{X}{R}", CardScriptConverter.applyTextCasing("{X}{R}"));
        // Mixed
        assertEquals("pay {X}, where X is the number of counters",
                CardScriptConverter.applyTextCasing("Pay {X}, where X is the number of counters"));
    }

    @Test
    void noCostCardOmitsManaCostLine() {
        ConvertedCard card = face("a/ancestral_vision.txt");
        assertNull(card.manaCost());
        assertFalse(OutputFormatter.formatCard(card).contains("mana cost:"));
    }

    @Test
    void textPropertyIncluded() {
        String output = OutputFormatter.formatCard(convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "Text:This is flavor text.", "Oracle:").faces().get(0));
        assertTrue(output.contains("text: this is flavor text."));
    }

    // --- US2: Complex card types ---

    @Test
    void planeswalkerAbilities() {
        ConvertedCard card = face("j/jace_beleren.txt");
        assertEquals("3", card.loyalty());
        var pw = abilitiesOfType(card, AbilityType.PLANESWALKER);
        assertEquals(3, pw.size());
        assertTrue(pw.get(0).formatLine().contains("[+2]:"));
        assertTrue(pw.get(1).formatLine().contains("[-1]:"));
        assertTrue(pw.get(2).formatLine().contains("[-10]:"));
        assertEquals(1, (int) pw.get(0).actionNumber());
        assertEquals(2, (int) pw.get(1).actionNumber());
        assertEquals(3, (int) pw.get(2).actionNumber());
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
        assertNull(chapters.get(0).actionNumber());
    }

    @Test
    void battleCardWithDefense() {
        MultiCard result = convertFromFile("i/invasion_of_kamigawa_rooftop_saboteurs.txt");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());

        ConvertedCard front = result.faces().get(0);
        assertEquals("invasion of kamigawa", front.name());
        assertEquals("4", front.defense());
        assertNull(front.powerToughness());
        assertTrue(OutputFormatter.formatCard(front).contains("defense: 4"));

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
        ConvertedCard card = face("l/lightning_bolt.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("CARDNAME deals 3 damage"));
        assertEquals(1, (int) spells.get(0).actionNumber());
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void spellWithAdditionalCost() {
        ConvertedCard card = face("a/abandon_hope.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.startsWith("additional cost:"));
        assertTrue(costLine.contains("discard X"));
        assertFalse(costLine.contains("as an additional cost to cast"));

        assertTrue(abilitiesOfType(card, AbilityType.SPELL).get(0).formatLine()
                .contains("look at target opponent's hand"));
    }

    @Test
    void cleaveEmittedAsAlternateCost() {
        ConvertedCard card = face("a/alchemists_gambit.txt");

        String altLine = abilitiesOfType(card, AbilityType.ALTERNATE_COST).get(0).formatLine();
        assertTrue(altLine.contains("cleave"));
        assertTrue(altLine.contains("{4}{U}{U}{R}"));
        assertFalse(altLine.matches(".*\\{R}\\d.*"), "No trailing digits after cost");

        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertFalse(spells.isEmpty());
        List<Integer> actionNumbers = spells.stream().map(AbilityLine::actionNumber).toList();
        for (int i = 0; i < actionNumbers.size() - 1; i++) {
            assertEquals(actionNumbers.get(i) + 1, (int) actionNumbers.get(i + 1));
        }
    }

    @Test
    void creatureWithAdditionalCost() {
        ConvertedCard card = face("a/abhorrent_oculus.txt");
        var costs = abilitiesOfType(card, AbilityType.ADDITIONAL_COST);
        assertEquals(1, costs.size());
        assertTrue(costs.get(0).formatLine().contains("exile"));
    }

    @Test
    void additionalCostFromRaiseCostStatic() {
        ConvertedCard card = face("a/aether_tide.txt");

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
        ConvertedCard card = face("a/aura_of_silence.txt");
        assertEquals(1, abilitiesOfType(card, AbilityType.STATIC).size());
        assertTrue(abilitiesOfType(card, AbilityType.STATIC).get(0).description().contains("cost {2} more to cast"));
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void optionalAdditionalCost() {
        ConvertedCard card = face("a/analyze_the_pollen.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("you may collect evidence 8"));
        assertFalse(costLine.contains("as an additional cost to cast"));
        assertEquals(0, countOfType(card, AbilityType.STATIC));
        assertCostsBeforeSpells(card);
    }

    @Test
    void alternateAdditionalCost() {
        ConvertedCard card = face("a/annihilating_glare.txt");

        String costLine = abilitiesOfType(card, AbilityType.ADDITIONAL_COST).get(0).formatLine();
        assertTrue(costLine.contains("sacrifice"));
        assertTrue(costLine.contains("or"));
        assertEquals(0, countOfType(card, AbilityType.KEYWORD_PASSIVE)
                + countOfType(card, AbilityType.KEYWORD_ACTIVE),
                "AlternateAdditionalCost should not appear as keyword");

        assertTrue(abilitiesOfType(card, AbilityType.SPELL).get(0).formatLine()
                .contains("destroy target creature or planeswalker"));
        assertCostsBeforeSpells(card);
    }

    @Test
    void pawprintCharmOptions() {
        ConvertedCard card = face("s/season_of_the_burrow.txt");
        assertEquals(1, abilitiesOfType(card, AbilityType.SPELL).size());

        var options = abilitiesOfType(card, AbilityType.OPTION);
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
        ConvertedCard card = face("a/artificer_class.txt");
        assertEquals("enchantment class", card.types());

        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        assertTrue(levels.get(0).formatLine().startsWith("level[1]:"));
        assertTrue(levels.get(0).formatLine().contains("first artifact spell"));
        assertFalse(levels.get(0).formatLine().contains("{1}{U}:"));
        assertTrue(levels.get(1).formatLine().startsWith("level[2]:"));
        assertTrue(levels.get(1).formatLine().contains("{1}{U}:"));
        assertTrue(levels.get(1).formatLine().contains("reveal cards"));
        assertTrue(levels.get(2).formatLine().startsWith("level[3]:"));
        assertTrue(levels.get(2).formatLine().contains("{5}{U}:"));
        assertTrue(levels.get(2).formatLine().contains("create a token"));

        assertEquals(0, countOfType(card, AbilityType.ACTIVATED));
        assertEquals(0, countOfType(card, AbilityType.STATIC));
    }

    @Test
    void classWithEtbReplacementLevel1() {
        ConvertedCard card = face("b/bard_class.txt");
        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        assertTrue(levels.get(0).formatLine().startsWith("level[1]:"));
        assertTrue(levels.get(0).formatLine().contains("legendary creatures"));
        assertNoRawEtbReplacementKeyword(card);
    }

    @Test
    void etbReplacementOnNonClassCard() {
        ConvertedCard card = face("f/flickering_ward.txt");
        assertNoRawEtbReplacementKeyword(card);
        assertTrue(abilitiesOfType(card, AbilityType.REPLACEMENT).stream()
                .anyMatch(r -> r.description().contains("choose a color")));
    }

    @Test
    void etbCounterWithDescriptionFallback() {
        ConvertedCard card = face("a/ambitious_dragonborn.txt");
        assertNoRawEtbReplacementKeyword(card);
        var replacements = abilitiesOfType(card, AbilityType.REPLACEMENT);
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).description().contains("CARDNAME enters with X +1/+1 counters"));
    }

    @Test
    void companionKeywordIncludesRestriction() {
        assertHasAbility(face("g/gyruda_doom_of_depths.txt"),
                AbilityType.KEYWORD_PASSIVE, "companion", "even mana value");
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
        assertHasAbility(face("d/dawns_truce.txt"), AbilityType.KEYWORD_PASSIVE, "gift a card");
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
        ConvertedCard card = convert(
                "Name:Test Convoke Spell", "ManaCost:3 W W", "Types:Sorcery",
                "K:Convoke", "A:SP$ Destroy | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "Oracle:").faces().get(0);
        assertHasAbility(card, AbilityType.COST_REDUCTION, "convoke");
    }

    // --- Cost ordering ---

    @Test
    void allCostTypesSortBeforeOtherAbilities() {
        // Whir of Invention: cost reduction should be first ability
        ConvertedCard whir = face("w/whir_of_invention.txt");
        assertTrue(whir.abilities().size() >= 2);
        assertEquals(AbilityType.COST_REDUCTION, whir.abilities().get(0).type());

        // Zephyrim: alternate cost + additional cost + keywords
        ConvertedCard zeph = face("z/zephyrim.txt");
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
        ConvertedCard card = face("a/aetherspouts.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("each attacking creature"));
    }

    @Test
    void activatedAbilityWithSubAbilityDescriptionSkipped() {
        // Arachnus Spinner's activated ability has SpellDescription only on the
        // sub-ability (DBChange), not on the main AB$ Pump. The sub-ability chain
        // walk must NOT cause this to be picked up as a spell — only spells should
        // walk the chain for SpellDescription.
        ConvertedCard card = face("a/arachnus_spinner.txt");
        assertEquals(0, countOfType(card, AbilityType.SPELL),
                "Activated ability with sub-ability description should not appear as spell");
    }

    // --- Charm sub-ability description walking ---

    @Test
    void charmWithSubAbilityDescription() {
        var options = abilitiesOfType(face("w/what_must_be_done.txt"), AbilityType.OPTION);
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
        var options = abilitiesOfType(convert(
                "Name:Chain Charm", "ManaCost:1 G", "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBChainTop,DBDirect",
                "SVar:DBChainTop:DB$ Pump | Defined$ Self | NumAtt$ 0 | NumDef$ 0 | SubAbility$ DBChainDesc",
                "SVar:DBChainDesc:DB$ Draw | NumCards$ 1 | SpellDescription$ Draw a card from the chain.",
                "SVar:DBDirect:DB$ GainLife | LifeAmount$ 3 | SpellDescription$ Gain 3 life.",
                "Oracle:").faces().get(0), AbilityType.OPTION);
        assertEquals(2, options.size());
        assertTrue(options.get(0).formatLine().contains("draw a card from the chain"));
        assertTrue(options.get(1).formatLine().contains("gain 3 life"));
    }

    // --- Helpers ---

    private void assertCostsBeforeSpells(ConvertedCard card) {
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

    private void assertNoRawEtbReplacementKeyword(ConvertedCard card) {
        long raw = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.KEYWORD_PASSIVE || a.type() == AbilityType.KEYWORD_ACTIVE)
                        && a.description().contains("etbreplacement"))
                .count();
        assertEquals(0, raw, "ETBReplacement should not appear as raw keyword");
    }
}
