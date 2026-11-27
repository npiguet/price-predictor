package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
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
        assertEquals("grizzly bears", card.name());
        assertTrue(card.manaCost().contains("{G}"));
        assertEquals("creature bear", card.types());
        assertEquals("2/2", card.powerToughness());
        assertTrue(card.abilities().isEmpty());
        assertNull(card.loyalty());
    }

    @Test
    void activatedAbility() {
        MultiCard elves = cards.convertFromFile("l/llanowar_elves.txt");
        CardFace card = elves.faces().get(0);
        String output = rendered(card);
        assertTrue(output.contains("activated[1]:"), output);
        assertTrue(output.contains("{T}"), output);
        assertTrue(output.contains("add {G}"), output);
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
        var pw = abilitiesOfType(card, AbilityType.PLANESWALKER);
        assertEquals(3, pw.size());
        assertTrue(lineOf(pw.get(0)).contains("[+2]:"));
        assertTrue(lineOf(pw.get(1)).contains("[-1]:"));
        assertTrue(lineOf(pw.get(2)).contains("[-10]:"));
        // Verify sequential action numbers in formatted output
        String output = rendered(card);
        assertTrue(output.contains("planeswalker[1]:"));
        assertTrue(output.contains("planeswalker[2]:"));
        assertTrue(output.contains("planeswalker[3]:"));
    }

    @Test
    void sagaChapterAbilities() {
        var chapters = abilitiesOfType(cards.face("t/the_eldest_reborn.txt"), AbilityType.CHAPTER);
        assertEquals(3, chapters.size());
        assertTrue(lineOf(chapters.get(0)).startsWith("chapter:"));
        assertTrue(lineOf(chapters.get(0)).contains("I \u2014"));
        assertTrue(lineOf(chapters.get(1)).contains("II \u2014"));
        assertTrue(lineOf(chapters.get(2)).contains("III \u2014"));
        assertTrue(lineOf(chapters.get(0)).contains("sacrifices"));
        assertFalse(lineOf(chapters.get(0)).contains("["), "Chapters should have no action number");
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

        var levels = abilitiesOfType(card, AbilityType.LEVEL);
        assertEquals(3, levels.size());
        // Class levels use ordinal() for fixed numbering in formatted output
        String output = rendered(card);
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
}
