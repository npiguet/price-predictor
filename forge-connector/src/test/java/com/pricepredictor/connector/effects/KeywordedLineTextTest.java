package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The text defects of keyed lines (finding 1.12): a repeated sentence, an
 * unexpanded stack template, the literal card name in a haunt line, an
 * unbracketed loyalty cost of 0, a double space after an ability word, a raw
 * selector, an empty option and an ability word in an additional cost.
 */
@ExtendWith(ForgeExtension.class)
class KeywordedLineTextTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private ConvertedScript convert(String relativePath) {
        return ConvertedScript.fromFile(parser, CARDS, relativePath);
    }

    private List<String> texts(String relativePath) {
        return convert(relativePath).texts();
    }

    private static int count(String text, String part) {
        return text.split(java.util.regex.Pattern.quote(part), -1).length - 1;
    }

    @Test
    void aSentenceAlreadyInTheDescriptionIsNotAppendedAgain() {
        String line = texts("k/krovikan_vampire.txt").stream()
                .filter(t -> t.startsWith("triggered")).findFirst().orElseThrow();
        assertEquals(1, count(line, "sacrifice it when you lose control of"), line);
    }

    @Test
    void aStackTemplateIsNotPrinted() {
        // Burrog Barrage: {C:...}; Arrow Storm: {TARGETED}.
        for (String card : List.of("b/burrog_barrage.txt", "a/arrow_storm.txt")) {
            for (String line : texts(card)) {
                assertFalse(line.contains("{C:") || line.contains("{TARGETED}"), line);
            }
        }
    }

    @Test
    void aHauntLineNamesTheCardAsCardname() {
        String line = texts("a/absolver_thrull.txt").stream()
                .filter(t -> t.contains("haunts dies")).findFirst().orElseThrow();
        assertFalse(line.contains("absolver thrull"), line);
        assertTrue(line.contains("CARDNAME"), line);
    }

    @Test
    void aLoyaltyCostOfZeroIsBracketed() {
        List<String> lines = texts("c/chandra_flamecaller.txt");
        assertTrue(lines.stream().anyMatch(t -> t.startsWith("planeswalker") && t.contains("[0]: ")), lines.toString());
        assertFalse(lines.stream().anyMatch(t -> t.startsWith("planeswalker") && t.contains("]: 0: ")), lines.toString());
    }

    @Test
    void anAbilityWordIsFollowedByOneSpace() {
        for (String line : texts("a/arni_brokenbrow.txt")) {
            assertFalse(line.contains("  "), line);
        }
    }

    @Test
    void aRawSelectorIsNotPrintedAndItsKeyIsDropped() {
        ConvertedScript converted = convert("s/say_its_name.txt");
        for (String line : converted.texts()) {
            assertFalse(line.contains("+youown") || line.contains("card.other"), line);
        }
        assertEquals(1, converted.texts().stream().filter(t -> t.startsWith("spell")).count());
        assertEquals(0, converted.texts().stream().filter(t -> t.startsWith("activated")).count());
        assertFalse(converted.sidecar().droppedKeys().isEmpty());
    }

    @Test
    void anEmptyOptionIsNotPrinted() {
        for (String line : texts("c/celebr_8000.txt")) {
            assertFalse(line.startsWith("option") && line.replaceAll("^[^:]*:", "").replaceAll("[, ]", "").isEmpty(), line);
        }
    }

    @Test
    void anAdditionalCostDoesNotCarryItsAbilityWord() {
        String line = texts("h/hierophant_bio_titan.txt").stream()
                .filter(t -> t.startsWith("additional cost")).findFirst().orElseThrow();
        assertFalse(line.contains("frenzied metabolism"), line);
    }
}
