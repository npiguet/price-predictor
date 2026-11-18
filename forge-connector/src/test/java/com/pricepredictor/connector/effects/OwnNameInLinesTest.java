package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A card's own name reads CARDNAME on every ability line, another card's name
 * stays, and no line keeps a run of blanks (finding 1.12).
 */
@ExtendWith(ForgeExtension.class)
class OwnNameInLinesTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private List<String> texts(String relativePath) {
        // The header keeps the card's name: only the ability lines are tested.
        return ConvertedScript.fromFile(parser, CARDS, relativePath).texts().stream()
                .filter(l -> !l.startsWith("name:")).toList();
    }

    private static boolean anyLine(List<String> lines, String part) {
        return lines.stream().anyMatch(l -> l.contains(part));
    }

    @Test
    void aKeywordLineNamingTheCardPrintsCardname() {
        List<String> lines = texts("x/xenosquirrels.txt");
        assertFalse(anyLine(lines, "xenosquirrels"), lines.toString());
        assertTrue(anyLine(lines, "from CARDNAME"), lines.toString());
    }

    @Test
    void aLoyaltyAbilityNamingTheCardPrintsCardname() {
        List<String> lines = texts("c/chandra_torch_of_defiance.txt");
        assertFalse(anyLine(lines, "chandra, torch of defiance"), lines.toString());
        assertTrue(anyLine(lines, "CARDNAME deals 2 damage to each opponent"), lines.toString());
    }

    @Test
    void thePartnersNameStaysAndNoBlankRunIsLeft() {
        List<String> lines = texts("b/blue_loyal_raptor.txt");
        assertTrue(anyLine(lines, "partner with"), lines.toString());
        assertFalse(anyLine(lines, "CARDNAME or"), lines.toString());
        for (String line : lines) assertFalse(line.contains("  "), line);
    }

    @Test
    void aBlankRunInASpellLineIsOneSpace() {
        for (String line : texts("l/light_up_the_night.txt")) {
            assertFalse(line.contains("  "), line);
        }
    }

    @Test
    void aCreatureTypeNamedCardKeepsItsLiteralName() {
        assertTrue(anyLine(texts("a/aurochs.txt"), "attacking aurochs"));
        assertTrue(anyLine(texts("a/assembly_worker.txt"), "target assembly-worker creature"));
    }

    @Test
    void anotherCardsLongerNameHoldingTheNameStays() {
        List<String> lines = texts("k/kookus.txt");
        assertTrue(anyLine(lines, "named keeper of kookus"), lines.toString());
        assertTrue(anyLine(lines, "CARDNAME deals 3 damage to you"), lines.toString());
    }

    @Test
    void aCardNamedByAKeywordActionKeepsTheAction() {
        // Exile: "Exile target nonwhite attacking creature. You gain life equal to its toughness."
        assertTrue(texts("e/exile.txt").contains(
                "spell[1]: exile target nonwhite attacking creature."
                        + " you gain life equal to its toughness."));
    }

    @Test
    void aCardNamedByAKeywordKeepsTheKeyword() {
        // Fear: "Enchanted creature has fear."
        assertTrue(texts("f/fear.txt").contains("static: enchanted creature has fear."));
    }

    @Test
    void aFaceNamedByAnOrdinaryWordKeepsTheWordInItsProse() {
        // Turn // Burn: "Until end of turn, target creature loses all abilities..."; "Burn deals 2 damage..."
        List<String> lines = texts("t/turn_burn.txt");
        assertTrue(anyLine(lines, "spell[2]: until end of turn, target creature loses all abilities"),
                lines.toString());
        assertTrue(lines.contains("spell[1]: CARDNAME deals 2 damage to any target."), lines.toString());
    }

    @Test
    void theNameWrittenInLowerCaseIsNoSelfReference() {
        // Six-Sided Die: "Roll a six-sided die."
        List<String> lines = texts("s/six_sided_die.txt");
        assertTrue(anyLine(lines, "roll a six-sided die."), lines.toString());
        assertFalse(anyLine(lines, "roll a CARDNAME"), lines.toString());
    }

    @Test
    void aTokensNameHoldingTheCardsNameStays() {
        // Stangg: "When Stangg enters, create Stangg Twin, a legendary 3/4 ... token."
        assertTrue(anyLine(texts("s/stangg.txt"), "when CARDNAME enters, create stangg twin, a legendary"));
    }
}
