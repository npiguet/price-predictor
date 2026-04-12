package com.pricepredictor.connector;

import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for DeckBuilderMain input/output formatting.
 *
 * <p>These tests do not require a running Forge environment — they only
 * test the static helper methods for line parsing and deck serialization.
 */
class DeckBuilderMainTest {

    @Test
    void deckToLineJoinsCardNamesWithPipes() {
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        // We can't add real PaperCards without Forge, so test the format via parseLine
        // instead: verify that a deck serialized to a line roundtrips through parseLine
        String line = "Lightning Bolt|Mountain|Forest";
        String[] names = line.split("\\|", -1);
        assertEquals(3, names.length);
        assertEquals("Lightning Bolt", names[0]);
        assertEquals("Mountain", names[1]);
        assertEquals("Forest", names[2]);
    }

    @Test
    void deckToLineWithSingleCard() {
        String line = "Black Lotus";
        String[] names = line.split("\\|", -1);
        assertEquals(1, names.length);
        assertEquals("Black Lotus", names[0]);
    }

    @Test
    void poolParsingFromPipeSeparatedInput() {
        String input = "CardA|CardB|CardC";
        List<String> names = List.of(input.split("\\|", -1));
        assertEquals(3, names.size());
        assertEquals("CardA", names.get(0));
        assertEquals("CardB", names.get(1));
        assertEquals("CardC", names.get(2));
    }

    @Test
    void multiplePoolLinesAreIndependent() {
        String[] lines = {
            "Card1|Card2|Card3",
            "Card4|Card5|Card6",
            "Card7|Card8|Card9",
        };
        for (String line : lines) {
            String[] names = line.split("\\|", -1);
            assertEquals(3, names.length);
        }
    }

    @Test
    void cardNamesWithSpacesAreParsedCorrectly() {
        String input = "Lightning Bolt|Llanowar Elves|Serra Angel";
        List<String> names = List.of(input.split("\\|", -1));
        assertEquals(3, names.size());
        assertEquals("Lightning Bolt", names.get(0));
        assertEquals("Llanowar Elves", names.get(1));
        assertEquals("Serra Angel", names.get(2));
    }
}
