package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MatchResultWriterTest {

    private static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    private static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    private static List<String> deck40(String prefix) {
        List<String> cards = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cards.add(prefix + "Card" + i);
        }
        return cards;
    }

    private static MatchResult sampleResult(List<String> deckA, List<String> deckB, String games, String play) {
        return new MatchResult(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                deckA, deckB, games, play, 47
        );
    }

    @Test
    void writesOneLinePerResult(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(sampleResult(deck40("A"), deck40("B"), "ABA", "BAB"));
        writer.write(sampleResult(deck40("C"), deck40("D"), "BB", "AA"));

        List<String> lines = Files.readAllLines(file);
        assertEquals(2, lines.size());
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

        String line = Files.readAllLines(file).get(0);
        String[] parts = line.split(";", -1);
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
    void appendsToExistingFile(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        Files.writeString(file, "an earlier run's line" + System.lineSeparator());
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(sampleResult(deck40("A"), deck40("B"), "AA", "BA"));

        List<String> lines = Files.readAllLines(file);
        assertEquals(2, lines.size());
        assertEquals("an earlier run's line", lines.get(0));
        assertTrue(lines.get(1).startsWith(DateTimeFormatter.ISO_INSTANT.format(TS) + ";"),
                lines.get(1));
    }

    @Test
    void invalidGamesStringRejected() {
        // Non-A/B characters are rejected regardless of length.
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "ABCD", "ABAB"));
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "AC", "BA"));
    }

    @Test
    void emptyGamesStringRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "", ""));
    }

    @Test
    void variedGameLengthsAccepted(@TempDir Path tmp) {
        // Match length is configurable via --best-of, so MatchResult must accept any
        // non-empty A/B string. Spot-check Bo3 (len 2-3), Bo5 (len 3-5), Bo7 (len 4-7),
        // and a large Bo9 example (len 5-9).
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "AA", "BA")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "ABA", "BAB")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "AABA", "BABA")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "ABABA", "BABAB")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "ABABAAB", "BABABAB")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "AAAAA", "BBBBB")));
        assertDoesNotThrow(() -> writer.write(sampleResult(deck40("A"), deck40("B"), "ABABABABA", "BABABABAB")));
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
