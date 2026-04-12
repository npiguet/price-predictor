package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ValidationMatchPlayerTest {

    @Test
    void parsesValidationMatchLine() {
        String line = "CardA|CardB|CardC;DeckD|DeckE|DeckF";
        ValidationMatchPlayer.ParsedMatch match = ValidationMatchPlayer.parseLine(line);
        assertEquals(List.of("CardA", "CardB", "CardC"), match.deckANames());
        assertEquals(List.of("DeckD", "DeckE", "DeckF"), match.deckBNames());
    }

    @Test
    void parseDeckWithSpaces() {
        String line = "Lightning Bolt|Mountain;Llanowar Elves|Forest";
        ValidationMatchPlayer.ParsedMatch match = ValidationMatchPlayer.parseLine(line);
        assertEquals("Lightning Bolt", match.deckANames().get(0));
        assertEquals("Mountain", match.deckANames().get(1));
        assertEquals("Llanowar Elves", match.deckBNames().get(0));
    }

    @Test
    void writesOutcomeFormat(@TempDir Path tmp) throws IOException {
        Path outcomes = tmp.resolve("outcomes.txt");
        ValidationMatchPlayer.appendOutcome(outcomes, 2, 1);
        List<String> lines = Files.readAllLines(outcomes);
        assertEquals(1, lines.size());
        assertEquals("2;1", lines.get(0));
    }

    @Test
    void appendsMultipleOutcomes(@TempDir Path tmp) throws IOException {
        Path outcomes = tmp.resolve("outcomes.txt");
        ValidationMatchPlayer.appendOutcome(outcomes, 2, 0);
        ValidationMatchPlayer.appendOutcome(outcomes, 1, 2);
        List<String> lines = Files.readAllLines(outcomes);
        assertEquals(2, lines.size());
        assertEquals("2;0", lines.get(0));
        assertEquals("1;2", lines.get(1));
    }

    @Test
    void crashRecoverySkipsCompletedMatches(@TempDir Path tmp) throws IOException {
        Path outcomes = tmp.resolve("outcomes.txt");
        // Simulate 2 already-completed matches
        Files.writeString(outcomes, "2;0\n1;2\n");

        long completed = ValidationMatchPlayer.countCompletedMatches(outcomes);
        assertEquals(2, completed);
    }

    @Test
    void crashRecoveryNoFileReturnsZero(@TempDir Path tmp) {
        Path outcomes = tmp.resolve("nonexistent.txt");
        long completed = ValidationMatchPlayer.countCompletedMatches(outcomes);
        assertEquals(0, completed);
    }
}
