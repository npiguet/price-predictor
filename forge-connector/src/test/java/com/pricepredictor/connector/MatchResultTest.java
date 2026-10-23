package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.HEADER;
import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.TS;
import static com.pricepredictor.connector.MatchFixtures.deck40;
import static com.pricepredictor.connector.MatchFixtures.games;
import static com.pricepredictor.connector.MatchFixtures.matchResult;
import static org.junit.jupiter.api.Assertions.*;

/**
 * The match-outcome line and what a {@link MatchResult} accepts. Appending is
 * {@link CorpusWriterTest}'s.
 */
class MatchResultTest {

    @Test
    void aLineIsTenFieldsInTheDocumentedOrder() {
        List<String> deckA = new ArrayList<>(List.of(
                "Lightning Bolt", "Snapcaster Mage", "Force of Will"));
        while (deckA.size() < 40) {
            deckA.add("Island");
        }
        List<String> deckB = deck40("B");

        String[] parts = matchResult(deckA, deckB, "ABA", "BAB").toLine().split(";", -1);
        assertEquals(10, parts.length, "Expected 10 semicolon-separated fields");
        assertEquals(DateTimeFormatter.ISO_INSTANT.format(TS), parts[0]);
        assertEquals(RUN_ID, parts[1]);
        assertEquals("RVR", parts[2]);
        assertEquals("forge-best", parts[3]);
        assertEquals("gen-2", parts[4]);
        // Names keep their spaces, order and duplicates; only the pipe separates them.
        assertEquals(String.join("|", deckA), parts[5]);
        assertEquals(String.join("|", deckB), parts[6]);
        assertEquals("ABA", parts[7]);
        assertEquals("BAB", parts[8]);
        assertEquals("47", parts[9]);
    }

    @Test
    void theSequencesAreOneCharacterPerGameInGameOrder() {
        // Match length is configurable via --best-of, so any non-empty number
        // of games is a row: Bo1 up to a long Bo9.
        List<List<String>> cases = List.of(
                List.of("A", "B"), List.of("AA", "BA"),
                List.of("ABABAAB", "BABABAB"), List.of("ABABABABA", "BABABABAB"));
        for (List<String> sequences : cases) {
            String[] parts = matchResult(deck40("A"), deck40("B"), sequences.get(0), sequences.get(1))
                    .toLine().split(";", -1);
            assertEquals(sequences.get(0), parts[7]);
            assertEquals(sequences.get(1), parts[8]);
        }
    }

    @Test
    void aMatchWithNoGamesIsRejected() {
        // GamePlayer skips draws and aborted games, so a match can end up
        // empty; a blank row must not reach the corpus.
        assertThrows(IllegalArgumentException.class, () ->
                new MatchResult(HEADER, deck40("A"), deck40("B"), List.of(), 0));
    }

    @Test
    void negativeDurationRejected() {
        assertThrows(IllegalArgumentException.class, () -> new MatchResult(
                HEADER, deck40("A"), deck40("B"), games("AA", "BA"), -1));
    }
}
