package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * In-memory index of a generated-decks file for self-play match generation.
 *
 * <p>File format: one deck per line, {@code LABEL;SET_CODE;Card1|Card2|...|Card40}.
 * The {@code LABEL} field is the generation-method tag set by
 * {@code build-decks --label} and used as the {@code method_A} / {@code method_B}
 * value when this deck is sampled into a self-play match.
 *
 * <p>Used by {@link SelfPlayMatchGenerator} to pick a random scorer-built deck A
 * each match, and to satisfy method 5 (pick another scorer-built deck B from the
 * same set as A, excluding A itself).
 */
public class GeneratedDecksIndex {

    /**
     * One parsed deck line.
     *
     * @param label     Generation-method tag (the line's first {@code ';'}-delimited field)
     * @param setCode   MTG set code (the second {@code ';'}-delimited field)
     * @param cardNames 40 card names (the pipe-separated tail)
     */
    public record GeneratedDeck(String label, String setCode, List<String> cardNames) {}

    private final List<GeneratedDeck> allDecks;
    private final Map<String, List<GeneratedDeck>> decksBySet;

    GeneratedDecksIndex(List<GeneratedDeck> decks) {
        this.allDecks = List.copyOf(decks);
        Map<String, List<GeneratedDeck>> grouped = new HashMap<>();
        for (GeneratedDeck deck : decks) {
            grouped.computeIfAbsent(deck.setCode(), k -> new ArrayList<>()).add(deck);
        }
        this.decksBySet = grouped;
    }

    /** Load and parse a generated-decks file. */
    public static GeneratedDecksIndex load(Path file) throws IOException {
        List<GeneratedDeck> decks = new ArrayList<>();
        for (String line : Files.readAllLines(file)) {
            if (line.isBlank()) continue;
            int firstSep = line.indexOf(';');
            int secondSep = firstSep < 0 ? -1 : line.indexOf(';', firstSep + 1);
            if (firstSep < 0 || secondSep < 0) {
                throw new IOException(
                        "Malformed generated-decks line (need 'LABEL;SET_CODE;Cards'): " + line);
            }
            String label = line.substring(0, firstSep);
            String setCode = line.substring(firstSep + 1, secondSep);
            String[] names = line.substring(secondSep + 1).split("\\|", -1);
            decks.add(new GeneratedDeck(label, setCode, List.of(names)));
        }
        if (decks.isEmpty()) {
            throw new IOException("Generated-decks file is empty: " + file);
        }
        return new GeneratedDecksIndex(decks);
    }

    public int size() {
        return allDecks.size();
    }

    /** Pick any random deck from the index. */
    public GeneratedDeck randomDeck(Random random) {
        return allDecks.get(random.nextInt(allDecks.size()));
    }

    /**
     * Pick a random deck from the same set as {@code exclude}, but never returning
     * {@code exclude} itself. Returns {@code null} if no other deck with this set
     * code exists in the index.
     */
    public GeneratedDeck randomDeckFromSet(String setCode, GeneratedDeck exclude, Random random) {
        List<GeneratedDeck> candidates = decksBySet.get(setCode);
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        if (candidates.size() == 1 && candidates.get(0) == exclude) {
            return null;
        }
        // Re-roll until we get a deck that is not the exclude reference.
        // Bounded by candidate count since at least one non-exclude exists.
        while (true) {
            GeneratedDeck pick = candidates.get(random.nextInt(candidates.size()));
            if (pick != exclude) {
                return pick;
            }
        }
    }
}
