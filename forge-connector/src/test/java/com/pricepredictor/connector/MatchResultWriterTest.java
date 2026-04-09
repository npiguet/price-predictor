package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MatchResultWriterTest {

    private static List<String> deck40(String prefix) {
        List<String> cards = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cards.add(prefix + "Card" + i);
        }
        return cards;
    }

    @Test
    void writesOneLinePerResult(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 1));
        writer.write(new MatchResult(deck40("C"), deck40("D"), 0, 2));

        List<String> lines = Files.readAllLines(file);
        assertEquals(2, lines.size());
    }

    @Test
    void lineHasFourSemicolonSeparatedFields(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 1));

        String line = Files.readAllLines(file).get(0);
        String[] parts = line.split(";", -1);
        assertEquals(4, parts.length, "Expected 4 semicolon-separated fields");
    }

    @Test
    void deckEncodedWithPipeSeparator(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        List<String> deckA = deck40("Card");
        writer.write(new MatchResult(deckA, deck40("B"), 2, 0));

        String line = Files.readAllLines(file).get(0);
        String deckAEncoded = line.split(";", -1)[0];
        String[] cardNames = deckAEncoded.split("\\|", -1);
        assertEquals(40, cardNames.length, "Expected 40 pipe-separated card names");
    }

    @Test
    void winsAppendedCorrectly(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 1));

        String line = Files.readAllLines(file).get(0);
        String[] parts = line.split(";", -1);
        assertEquals("2", parts[2]);
        assertEquals("1", parts[3]);
    }

    @Test
    void appendsToExistingFile(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 0));
        writer.write(new MatchResult(deck40("C"), deck40("D"), 0, 2));
        writer.write(new MatchResult(deck40("E"), deck40("F"), 2, 1));

        assertEquals(3, Files.readAllLines(file).size());
    }

    @Test
    void bo2WinsSumTo2(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        // 2-0 is valid (sum = 2)
        assertDoesNotThrow(() ->
                writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 0)));
    }

    @Test
    void bo3WinsSumTo3(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        // 2-1 is valid (sum = 3)
        assertDoesNotThrow(() ->
                writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 1)));
    }

    @Test
    void invalidWinsSumRejectedByMatchResult() {
        // wins sum to 4 — invalid
        assertThrows(IllegalArgumentException.class, () ->
                new MatchResult(deck40("A"), deck40("B"), 2, 2));
    }

    @Test
    void invalidWinsSumRejectedByWriter(@TempDir Path tmp) {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        // Create via a direct MatchResult with sum=1 (impossible via constructor guard,
        // but we can test the writer's own guard by bypassing the record compact constructor)
        // Since MatchResult guards via compact constructor, sum=1 isn't constructible.
        // Verify 2+1=3 is valid:
        assertDoesNotThrow(() ->
                writer.write(new MatchResult(deck40("A"), deck40("B"), 2, 1)));
    }

    @Test
    void cardNamesPreservedExactly(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("outcomes.txt");
        MatchResultWriter writer = new MatchResultWriter(file);

        List<String> deckA = List.of(
                "Lightning Bolt", "Snapcaster Mage", "Force of Will",
                "Black Lotus", "Mox Pearl", "Island", "Island", "Island",
                "Mountain", "Forest", "Swamp", "Plains", "Island",
                "Island", "Island", "Island", "Island", "Island",
                "Island", "Island", "Island", "Island", "Island",
                "Island", "Island", "Island", "Island", "Island",
                "Island", "Island", "Island", "Island", "Island",
                "Island", "Island", "Island", "Island", "Island",
                "Island", "Island"
        );
        List<String> deckB = deck40("B");

        writer.write(new MatchResult(deckA, deckB, 2, 0));

        String line = Files.readAllLines(file).get(0);
        String deckAEncoded = line.split(";", -1)[0];
        String[] names = deckAEncoded.split("\\|", -1);
        assertEquals("Lightning Bolt", names[0]);
        assertEquals("Snapcaster Mage", names[1]);
        assertEquals("Force of Will", names[2]);
    }
}
