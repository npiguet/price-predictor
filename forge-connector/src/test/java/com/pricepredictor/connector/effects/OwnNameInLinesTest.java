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
}
