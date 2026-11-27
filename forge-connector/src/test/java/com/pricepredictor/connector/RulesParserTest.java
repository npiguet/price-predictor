package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertContainsAll;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertNoRawEtbReplacementKeyword;
import static com.pricepredictor.connector.ConvertedTextAssertions.assertOccursOnce;
import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.countOfType;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's conversion of each ability type and card type, and the text it prints for them. */
@ExtendWith(ForgeExtension.class)
class RulesParserTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void vanillaCreature() {
        CardFace card = cards.face("g/grizzly_bears.txt");
        assertEquals(new FaceHeader("grizzly bears", "creature bear", "2/2", null), FaceHeader.of(card));
        assertTrue(card.manaCost().contains("{G}"));
        assertTrue(card.abilities().isEmpty());
    }

    /** A face's name, type line, power and toughness, and loyalty. */
    private record FaceHeader(String name, String types, String powerToughness, String loyalty) {

        static FaceHeader of(CardFace face) {
            return new FaceHeader(face.name(), face.types(), face.powerToughness(), face.loyalty());
        }
    }

    @Test
    void activatedAbility() {
        MultiCard elves = cards.convertFromFile("l/llanowar_elves.txt");
        CardFace card = elves.faces().get(0);
        assertContainsAll(rendered(card), "activated[1]:", "{T}", "add {G}");
        // The text is lowercased, the mana symbols kept as written.
        assertTrue(rendered(elves).contains("llanowar elves"), rendered(elves));

        // A simple activated ability (no sub-ability SpellDescription) is one
        // line, and the chain walk must not duplicate its description text.
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertOccursOnce(activated.get(0).descriptionText(), "add");
    }

    @Test
    void triggeredAbility() {
        var triggered = abilitiesOfType(cards.face("t/thragtusk.txt"), AbilityType.TRIGGERED);
        assertFalse(triggered.isEmpty());
        String line = lineOf(triggered.get(0));
        assertTrue(line.startsWith("triggered:"));
        assertTrue(line.contains("CARDNAME"));
        assertTrue(line.contains("gain 5 life"));
    }

    @Test
    void staticAbility() {
        var statics = abilitiesOfType(cards.face("b/blood_moon.txt"), AbilityType.STATIC);
        assertEquals(1, statics.size());
        assertTrue(lineOf(statics.get(0)).contains("nonbasic lands are mountains"));
    }

    @Test
    void replacementEffect() {
        var replacements = abilitiesOfType(cards.face("r/rest_in_peace.txt"), AbilityType.REPLACEMENT);
        assertFalse(replacements.isEmpty());
        assertTrue(replacements.stream().anyMatch(r -> lineOf(r).contains("exile it instead")));
    }

    @Test
    void staticTriggerEmittedAsReplacement() {
        // Domesticated Mammoth has a trigger with Static$ True ("as this enters").
        // In MTG rules these are replacement effects, not triggered abilities.
        CardFace card = cards.face("d/domesticated_mammoth.txt");
        var replacements = abilitiesOfType(card, AbilityType.REPLACEMENT);
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).descriptionText().contains("token copy of pacifism"));
        assertEquals(0, countOfType(card, AbilityType.TRIGGERED));
    }

    @Test
    void secondaryTriggerNotSkipped() {
        // Decorated Champion has a trigger marked Secondary$ True in Forge.
        // It should still be emitted since it's the card's actual ability text.
        CardFace card = cards.face("d/decorated_champion.txt");
        var triggered = abilitiesOfType(card, AbilityType.TRIGGERED);
        assertEquals(1, triggered.size());
        assertTrue(triggered.get(0).descriptionText().contains("put a +1/+1 counter"));
    }

    @Test
    void engineMetadataExcluded() {
        String output = rendered(cards.convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AI:RemoveDeck:Random", "DeckHints:Ability$LifeGain",
                "SVar:AnotherSVar:DB$ Something", "Oracle:"));
        assertFalse(output.contains("AI:"));
        assertFalse(output.contains("DeckHints"));
        assertFalse(output.contains("SVar"));
    }

    @Test
    void actionCounterIncrementsAcrossMixedTypes() {
        CardFace card = cards.convert(
                "Name:Versatile Card", "ManaCost:2 W", "Types:Creature Human", "PT:2/2",
                "K:Cycling:2", "A:AB$ GainLife | Cost$ T | LifeAmount$ 1 | SpellDescription$ Gain 1 life.",
                "Oracle:").faces().get(0);
        // Verify sequential action numbers in formatted output
        String output = rendered(card);
        assertTrue(output.contains("[1]:"), "Should have action number 1: " + output);
        assertTrue(output.contains("[2]:"), "Should have action number 2: " + output);
        assertTrue(output.indexOf("[1]:") < output.indexOf("[2]:"),
                "Action numbers should be sequential in output: " + output);
    }

    @Test
    void applyCasingLowercasesTheTextButKeepsVariableXAndManaSymbols() {
        List<String> texts = List.of(
                // Standalone X preserved
                "-X: Deal X damage", "+X/+0 until end of turn", "Where X is the number",
                // X inside words lowercased
                "Exile target creature", "Next end step", "Tax each opponent",
                // Braces
                "{X}{R}",
                // Mixed
                "Pay {X}, where X is the number of counters");
        assertEquals(List.of(
                        "-X: deal X damage", "+X/+0 until end of turn", "where X is the number",
                        "exile target creature", "next end step", "tax each opponent",
                        "{X}{R}",
                        "pay {X}, where X is the number of counters"),
                texts.stream().map(AbilityDescription::applyCasing).toList());
    }

    @Test
    void noCostCardOmitsManaCostLine() {
        CardFace card = cards.face("a/ancestral_vision.txt");
        assertNull(card.manaCost());
        assertFalse(rendered(card).contains("mana cost:"));
    }

    @Test
    void textPropertyIncluded() {
        String output = rendered(cards.convert(
                "Name:Test Card", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "Text:This is flavor text.", "Oracle:").faces().get(0));
        assertTrue(output.contains("text: this is flavor text."));
    }

    @Test
    void planeswalkerAbilities() {
        CardFace card = cards.face("j/jace_beleren.txt");
        assertEquals("3", card.loyalty());
        assertEachHolds(linesOf(abilitiesOfType(card, AbilityType.PLANESWALKER)),
                List.of(List.of("[+2]:"), List.of("[-1]:"), List.of("[-10]:")));
        // Verify sequential action numbers in formatted output
        assertContainsAll(rendered(card), "planeswalker[1]:", "planeswalker[2]:", "planeswalker[3]:");
    }

    @Test
    void sagaChapterAbilities() {
        var chapters = linesOf(abilitiesOfType(cards.face("t/the_eldest_reborn.txt"), AbilityType.CHAPTER));
        assertEachHolds(chapters, List.of(
                List.of("I \u2014", "sacrifices"), List.of("II \u2014"), List.of("III \u2014")));
        String first = chapters.get(0);
        assertTrue(first.startsWith("chapter:"), first);
        assertFalse(first.contains("["), "Chapters should have no action number");
    }

    @Test
    void spellEffect() {
        CardFace card = cards.face("l/lightning_bolt.txt");
        var spells = abilitiesOfType(card, AbilityType.SPELL);
        assertEquals(1, spells.size());
        assertTrue(lineOf(spells.get(0)).contains("CARDNAME deals 3 damage"));
        // Verify action number in formatted output
        String output = rendered(card);
        assertTrue(output.contains("spell[1]:"));
        assertEquals(0, countOfType(card, AbilityType.ADDITIONAL_COST));
    }

    @Test
    void classEnchantmentLevels() {
        CardFace card = cards.face("a/artificer_class.txt");
        assertEquals("enchantment class", card.types());
        // Class levels use ordinal() for fixed numbering in formatted output
        assertContainsAll(rendered(card), "level[1]:", "level[2]:", "level[3]:");

        List<String> levels = abilitiesOfType(card, AbilityType.LEVEL).stream().map(Ability::descriptionText).toList();
        assertEachHolds(levels, List.of(
                List.of("first artifact spell"), List.of("{1}{U}:", "reveal cards"), List.of("{5}{U}:", "create a token")));
        assertFalse(levels.get(0).contains("{1}{U}:"), levels.get(0));
        assertEquals(List.of(0L, 0L),
                List.of(countOfType(card, AbilityType.ACTIVATED), countOfType(card, AbilityType.STATIC)),
                "activated and static lines: " + card.abilities());
    }

    @Test
    void classWithEtbReplacementLevel1() {
        CardFace card = cards.face("b/bard_class.txt");
        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        String output = rendered(card);
        assertTrue(output.contains("level[1]:"), "Level 1 in output: " + output);
        assertTrue(levels.get(0).descriptionText().contains("legendary creatures"));
        assertNoRawEtbReplacementKeyword(card);
    }

    @Test
    void sagaChapterWithHeaderOnlyTriggerDescriptionResolvesContent() {
        // ballad_of_the_black_flag: TriggerDescription is just "I, II, III —" with
        // the effect on the execute SVar.  Chapter lines must include the effect text.
        CardFace card = cards.face("b/ballad_of_the_black_flag.txt");
        var chapters = abilitiesOfType(card, AbilityType.CHAPTER);
        assertFalse(chapters.isEmpty(), "Should have CHAPTER abilities: " + card.abilities());
        // At least one chapter must have non-empty effect text after the roman numeral header
        assertTrue(chapters.stream().anyMatch(c -> {
            String desc = c.descriptionText();
            int dash = desc.indexOf('\u2014');
            return dash >= 0 && !desc.substring(dash + 1).trim().isEmpty();
        }), "At least one chapter should have effect text after em-dash: " + chapters);
    }

    @Test
    void silvosActivatedAbilityUsesCardnamePlaceholder() {
        // SpellDescription$ Regenerate CARDNAME. — Java side already correct.
        var card = cards.face("s/silvos_rogue_elemental.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertTrue(activated.get(0).descriptionText().contains("regenerate"));
        assertFalse(activated.get(0).descriptionText().toLowerCase().contains("silvos"),
                    "Card name must be replaced, not kept literal: " + activated);
    }

    @Test
    void nicknameReplacedWithCardname() {
        // NICKNAME is a Forge alias for the card name — must appear as CARDNAME in output.
        CardFace card = cards.convert(
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

    @ParameterizedTest
    @CsvSource({"c/celestine_cave_witch.txt", "g/goblin_polka_band.txt"})
    void developerNoteStrippedFromTextField(String file) {
        // Text:[Developer's note: …] is not oracle content; must not be emitted.
        var card = cards.face(file);
        assertNull(card.text(), "Developer's note should produce null text field, got: " + card.text());
    }

    /**
     * A Class level adding a trait with no text renders the text it does have.
     *
     * <p>Advanced Floral Invocations' third level adds a static and a
     * {@code Static$ True} helper trigger with no {@code TriggerDescription$}.
     * Forge joins the level's descriptions unchecked, so its own level static
     * reads {@code null} followed by the static's text on a second line; the
     * level must print as one line carrying only the static's text.
     */
    @Test
    void aClassLevelAddingAnUndescribedTraitPrintsOnlyTheDescribedText() {
        String output = rendered(cards.face("a/advanced_floral_invocations.txt"));

        assertFalse(output.contains("null"), output);
        assertTrue(output.contains("level[3]: {4}{B/G}{B/G}: you may play lands and cast"
                + " creature spells from your graveyard."), output);
        assertEquals(6, output.split("\n").length,
                "name, mana cost, types and three level lines: " + output);
    }

    /**
     * A description Forge joined with {@code \r\n} renders as clean lines.
     *
     * <p>Ranger Class's third level adds two statics, and Forge joins their
     * descriptions with a carriage return and a line feed. The level is one
     * line, the break read as a space, and no carriage return is left in it.
     */
    @Test
    void aDescriptionJoinedWithACarriageReturnRendersWithoutOne() {
        String output = rendered(cards.face("r/ranger_class.txt"));

        assertFalse(output.contains("\r"), output);
        assertTrue(output.contains("level[3]: {3}{G}: you may look at the top card of your"
                + " library any time. you may cast creature spells from the top of your"
                + " library."), output);
    }

    /** The line the converter writes for each of {@code abilities}, in order. */
    private static List<String> linesOf(List<Ability> abilities) {
        return abilities.stream().map(AbilityRendering::lineOf).toList();
    }

    /** {@code texts} are one per entry of {@code expectedParts}, in order, each holding every part of its entry. */
    private static void assertEachHolds(List<String> texts, List<List<String>> expectedParts) {
        assertEquals(expectedParts.size(), texts.size(), texts.toString());
        for (int i = 0; i < texts.size(); i++) {
            assertContainsAll(texts.get(i), expectedParts.get(i).toArray(String[]::new));
        }
    }
}
