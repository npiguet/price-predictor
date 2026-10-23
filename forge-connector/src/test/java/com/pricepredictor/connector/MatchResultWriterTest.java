package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.TS;
import static com.pricepredictor.connector.MatchFixtures.deck40;
import static org.junit.jupiter.api.Assertions.*;

/**
 * The match-outcome line and what a {@link MatchResult} accepts. Appending is
 * {@link LineAppenderTest}'s.
 */
class MatchResultWriterTest {

    private static MatchResult sampleResult(List<String> deckA, List<String> deckB, String games, String play) {
        return new MatchResult(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                deckA, deckB, games, play, 47
        );
    }

    @Test
    void aLineIsTenFieldsInTheDocumentedOrder(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);
        List<String> deckA = new ArrayList<>(List.of(
                "Lightning Bolt", "Snapcaster Mage", "Force of Will"));
        while (deckA.size() < 40) {
            deckA.add("Island");
        }
        List<String> deckB = deck40("B");

        writer.write(sampleResult(deckA, deckB, "ABA", "BAB"));

        List<String> lines = Files.readAllLines(file);
        assertEquals(1, lines.size());
        String[] parts = lines.get(0).split(";", -1);
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

    /** The directory exists once the writer does, before its first match is written. */
    @Test
    void aWriterNeedsNoExistingDirectory(@TempDir Path tmp) {
        Path file = tmp.resolve("draft").resolve("draft-games.txt");

        new MatchResultWriter(file);

        assertTrue(Files.isDirectory(file.getParent()));
    }

    @Test
    void invalidGamesOrPlayCharacterRejected() {
        // Non-A/B characters are rejected in either sequence, each by its own check.
        IllegalArgumentException games = assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "ABCD", "ABAB"));
        assertTrue(games.getMessage().startsWith("games must contain"), games.getMessage());
        IllegalArgumentException play = assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "AB", "BC"));
        assertTrue(play.getMessage().startsWith("play must contain"), play.getMessage());
    }

    @Test
    void emptyGamesStringRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "", ""));
    }

    @Test
    void variedGameLengthsAccepted() {
        // Match length is configurable via --best-of, so MatchResult must accept any
        // non-empty A/B string. Spot-check Bo3 (len 2-3), Bo5 (len 3-5), Bo7 (len 4-7),
        // and a large Bo9 example (len 5-9).
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "AA", "BA"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "ABA", "BAB"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "AABA", "BABA"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "ABABA", "BABAB"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "ABABAAB", "BABABAB"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "AAAAA", "BBBBB"));
        assertDoesNotThrow(() -> sampleResult(deck40("A"), deck40("B"), "ABABABABA", "BABABABAB"));
    }

    @Test
    void mismatchedGamesAndPlayLengthRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "AA", "BAB"));
    }

    @Test
    void negativeDurationRejected() {
        assertThrows(IllegalArgumentException.class, () -> new MatchResult(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                deck40("A"), deck40("B"), "AA", "BA", -1
        ));
    }
}
