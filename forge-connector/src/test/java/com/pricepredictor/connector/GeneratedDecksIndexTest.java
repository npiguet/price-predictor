package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure unit tests for {@link GeneratedDecksIndex} — no Forge dependency.
 *
 * <p>File format under test: {@code LABEL;SET_CODE;Card1|Card2|...|CardN}.
 */
class GeneratedDecksIndexTest {

    private static Path writeDecks(Path dir, List<String> lines) throws IOException {
        Path file = dir.resolve("generated-decks.txt");
        Files.writeString(file, String.join("\n", lines) + "\n");
        return file;
    }

    private static String deckLine(String label, String setCode, String prefix, int n) {
        StringBuilder sb = new StringBuilder(label).append(';').append(setCode).append(';');
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append('|');
            sb.append(prefix).append('_').append(i);
        }
        return sb.toString();
    }

    @Test
    void loadParsesLabelSetCodeAndCardNames(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(
                "gen-2;MH3;A|B|C",
                "gen-3;BLB;X|Y|Z|W"
        ));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);

        assertEquals(2, index.size());
        Set<String> seenSets = new HashSet<>();
        Set<String> seenLabels = new HashSet<>();
        Random rng = new Random(0);
        for (int i = 0; i < 50; i++) {
            GeneratedDecksIndex.GeneratedDeck pick = index.randomDeck(rng);
            seenSets.add(pick.setCode());
            seenLabels.add(pick.label());
        }
        assertEquals(Set.of("MH3", "BLB"), seenSets);
        assertEquals(Set.of("gen-2", "gen-3"), seenLabels);
    }

    @Test
    void loadIgnoresBlankLines(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(
                "gen-2;MH3;A|B",
                "",
                "gen-2;MH3;C|D",
                "   "
        ));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);

        assertEquals(2, index.size());
    }

    @Test
    void loadEmptyFileThrows(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("empty.txt");
        Files.writeString(file, "");

        assertThrows(IOException.class, () -> GeneratedDecksIndex.load(file));
    }

    @Test
    void loadMissingSecondSeparatorThrows(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of("gen-2;onlyOneSemicolon"));

        assertThrows(IOException.class, () -> GeneratedDecksIndex.load(file));
    }

    @Test
    void loadMissingAnySeparatorThrows(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of("no-semicolon-here"));

        assertThrows(IOException.class, () -> GeneratedDecksIndex.load(file));
    }

    @Test
    void randomDeckPicksFromAllDecks(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(
                deckLine("gen-2", "MH3", "mh3", 40),
                deckLine("gen-2", "BLB", "blb", 40),
                deckLine("gen-2", "RVR", "rvr", 40)
        ));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);
        Random rng = new Random(0);

        Set<String> seen = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            seen.add(index.randomDeck(rng).setCode());
        }

        assertEquals(Set.of("MH3", "BLB", "RVR"), seen);
    }

    @Test
    void randomDeckFromSetReturnsOnlyMatchingSet(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(
                deckLine("gen-2", "MH3", "mh3a", 40),
                deckLine("gen-2", "MH3", "mh3b", 40),
                deckLine("gen-2", "BLB", "blb", 40)
        ));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);
        Random rng = new Random(42);

        for (int i = 0; i < 100; i++) {
            GeneratedDecksIndex.GeneratedDeck pick = index.randomDeckFromSet("MH3", null, rng);
            assertNotNull(pick);
            assertEquals("MH3", pick.setCode());
        }
    }

    @Test
    void randomDeckFromSetExcludesGivenDeck(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(
                deckLine("gen-2", "MH3", "mh3a", 40),
                deckLine("gen-2", "MH3", "mh3b", 40)
        ));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);
        Random rng = new Random(7);

        GeneratedDecksIndex.GeneratedDeck a = index.randomDeck(rng);

        for (int i = 0; i < 50; i++) {
            GeneratedDecksIndex.GeneratedDeck b = index.randomDeckFromSet("MH3", a, rng);
            assertNotNull(b);
            assertNotSame(a, b, "exclude target must never be returned");
        }
    }

    @Test
    void randomDeckFromSetReturnsNullWhenSetAbsent(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(deckLine("gen-2", "MH3", "mh3", 40)));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);

        assertNull(index.randomDeckFromSet("XYZ", null, new Random(0)));
    }

    @Test
    void randomDeckFromSetReturnsNullWhenOnlyExcludeMatches(@TempDir Path tmp) throws IOException {
        Path file = writeDecks(tmp, List.of(deckLine("gen-2", "MH3", "only", 40)));

        GeneratedDecksIndex index = GeneratedDecksIndex.load(file);
        GeneratedDecksIndex.GeneratedDeck only = index.randomDeck(new Random(0));

        assertNull(index.randomDeckFromSet("MH3", only, new Random(0)));
    }
}
