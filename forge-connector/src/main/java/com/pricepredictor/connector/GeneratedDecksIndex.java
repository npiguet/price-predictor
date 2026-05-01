package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
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
 * value when this deck is sampled into a match.
 *
 * <p>Used by {@link MatchGenerator}'s side-A / side-B sampling paths: an index
 * supplied via {@code -Dside.a.decks.file} drives {@link #randomDeck} (deck A);
 * an index supplied via {@code -Dside.b.decks.file} drives
 * {@link #randomDeckFromSet} (deck B, filtered to deck A's set code with
 * mirror-match exclusion).
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
    /**
     * Per-deck sorted card-name list, indexed by the deck's identity in
     * {@link #allDecks}. Pre-computed at load time so the same-content
     * mirror-match check in {@link #randomDeckFromSet(String, List, Random)}
     * is O(40) per candidate instead of O(40 log 40).
     */
    private final Map<GeneratedDeck, List<String>> sortedCardsByDeck;

    GeneratedDecksIndex(List<GeneratedDeck> decks) {
        this.allDecks = List.copyOf(decks);
        Map<String, List<GeneratedDeck>> grouped = new HashMap<>();
        Map<GeneratedDeck, List<String>> sorted = new HashMap<>();
        for (GeneratedDeck deck : decks) {
            grouped.computeIfAbsent(deck.setCode(), k -> new ArrayList<>()).add(deck);
            List<String> sortedCards = new ArrayList<>(deck.cardNames());
            Collections.sort(sortedCards);
            sorted.put(deck, List.copyOf(sortedCards));
        }
        this.decksBySet = grouped;
        this.sortedCardsByDeck = sorted;
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
     * Pick a random deck from the given set whose card list is not a
     * permutation of {@code excludeCards}. Returns {@code null} if no
     * matching deck exists. The exclusion uses content equality (multiset
     * over card names) rather than reference equality, so a deck loaded
     * from this index is treated as a mirror of {@code excludeCards} even
     * if it was sampled from a different index instance with the same
     * content — this is the case when {@code --side-a-decks} and
     * {@code --side-b-decks} point at overlapping files.
     *
     * @param setCode      MTG set code to filter on
     * @param excludeCards Card-name list to exclude (e.g. deck A's cards);
     *                     order-insensitive
     * @param random       Source of randomness
     */
    public GeneratedDeck randomDeckFromSet(
            String setCode, List<String> excludeCards, Random random) {
        List<GeneratedDeck> candidates = decksBySet.get(setCode);
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        List<String> excludeSorted = new ArrayList<>(excludeCards);
        Collections.sort(excludeSorted);
        // Filter to non-mirror candidates first, then sample uniformly. This
        // keeps the sample distribution flat when there are many mirrors,
        // unlike rejection sampling.
        List<GeneratedDeck> nonMirror = new ArrayList<>(candidates.size());
        for (GeneratedDeck c : candidates) {
            if (!sortedCardsByDeck.get(c).equals(excludeSorted)) {
                nonMirror.add(c);
            }
        }
        if (nonMirror.isEmpty()) {
            return null;
        }
        return nonMirror.get(random.nextInt(nonMirror.size()));
    }
}
