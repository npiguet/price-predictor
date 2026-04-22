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
    void lineHasTenSemicolonSeparatedFields(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(sampleResult(deck40("A"), deck40("B"), "ABA", "BAB"));

        String line = Files.readAllLines(file).get(0);
        String[] parts = line.split(";", -1);
        assertEquals(10, parts.length, "Expected 10 semicolon-separated fields");
    }

    @Test
    void deckEncodedWithPipeSeparator(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        List<String> deckA = deck40("Card");
        writer.write(sampleResult(deckA, deck40("B"), "AA", "BA"));

        String line = Files.readAllLines(file).get(0);
        String deckAEncoded = line.split(";", -1)[5];
        String[] cardNames = deckAEncoded.split("\\|", -1);
        assertEquals(40, cardNames.length, "Expected 40 pipe-separated card names");
    }

    @Test
    void fieldsAppearInExpectedOrder(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(sampleResult(deck40("A"), deck40("B"), "ABA", "BAB"));

        String line = Files.readAllLines(file).get(0);
        String[] parts = line.split(";", -1);
        assertEquals(DateTimeFormatter.ISO_INSTANT.format(TS), parts[0]);
        assertEquals(RUN_ID, parts[1]);
        assertEquals("RVR", parts[2]);
        assertEquals("forge-best", parts[3]);
        assertEquals("gen-2", parts[4]);
        // parts[5] = deckA, parts[6] = deckB
        assertEquals("ABA", parts[7]);
        assertEquals("BAB", parts[8]);
        assertEquals("47", parts[9]);
    }

    @Test
    void appendsToExistingFile(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(sampleResult(deck40("A"), deck40("B"), "AA", "BA"));
        writer.write(sampleResult(deck40("C"), deck40("D"), "BB", "AB"));
        writer.write(sampleResult(deck40("E"), deck40("F"), "ABA", "BAB"));

        assertEquals(3, Files.readAllLines(file).size());
    }

    @Test
    void twoGameSweepAccepted(@TempDir Path tmp) {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        assertDoesNotThrow(() ->
                writer.write(sampleResult(deck40("A"), deck40("B"), "AA", "BA")));
    }

    @Test
    void threeGameMatchAccepted(@TempDir Path tmp) {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        assertDoesNotThrow(() ->
                writer.write(sampleResult(deck40("A"), deck40("B"), "ABA", "BAB")));
    }

    @Test
    void winsAandWinsBDerivedFromGamesString() {
        MatchResult r = sampleResult(deck40("A"), deck40("B"), "ABA", "BAB");
        assertEquals(2, r.winsA());
        assertEquals(1, r.winsB());

        MatchResult sweep = sampleResult(deck40("A"), deck40("B"), "BB", "AA");
        assertEquals(0, sweep.winsA());
        assertEquals(2, sweep.winsB());
    }

    @Test
    void invalidGamesStringRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "A", "A"));
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "ABCD", "ABAB"));
        assertThrows(IllegalArgumentException.class, () ->
                sampleResult(deck40("A"), deck40("B"), "AC", "BA"));
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

    @Test
    void cardNamesPreservedExactly(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        List<String> deckA = new ArrayList<>();
        deckA.add("Lightning Bolt");
        deckA.add("Snapcaster Mage");
        deckA.add("Force of Will");
        while (deckA.size() < 40) {
            deckA.add("Island");
        }

        writer.write(sampleResult(deckA, deck40("B"), "AA", "BA"));

        String line = Files.readAllLines(file).get(0);
        String deckAEncoded = line.split(";", -1)[5];
        String[] names = deckAEncoded.split("\\|", -1);
        assertEquals("Lightning Bolt", names[0]);
        assertEquals("Snapcaster Mage", names[1]);
        assertEquals("Force of Will", names[2]);
    }
}
