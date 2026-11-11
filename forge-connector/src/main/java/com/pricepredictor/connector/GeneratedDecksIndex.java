package com.pricepredictor.connector;

import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.Multiset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

/**
 * In-memory index of a generated-decks file for self-play match generation.
 *
 * <p>File format: one deck per line, {@code LABEL;SET_CODE;Card1|Card2|...|Card40},
 * and a steered variant round's deck adds the rule it is aimed at as a fourth
 * field, {@code LABEL;SET_CODE;Card1|...|Card40;RULE} (FR-058d).
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
     * @param cardNames 40 card names (the pipe-separated third field)
     * @param rule      the rule a steered deck is aimed at (the fourth field),
     *                  or null for a deck without one
     */
    public record GeneratedDeck(
            String label, String setCode, List<String> cardNames, String rule) {

        /** A deck aimed at no rule. */
        public GeneratedDeck(String label, String setCode, List<String> cardNames) {
            this(label, setCode, cardNames, null);
        }

        /**
         * One {@code LABEL;SET_CODE;Card1|...} line, or a steered deck's
         * {@code LABEL;SET_CODE;Card1|...;RULE}.
         *
         * <p>Split into at most four fields: a card name holds no {@code ;},
         * so a fourth field is the rule, never part of the last card's name.
         * An empty fourth field is no rule.
         *
         * @throws IOException when the line lacks either separator
         */
        static GeneratedDeck parse(String line) throws IOException {
            String[] fields = line.split(CardNameList.FIELD_SEPARATOR, 4);
            if (fields.length < 3) {
                throw new IOException(
                        "Malformed generated-decks line (need 'LABEL;SET_CODE;Cards[;RULE]'): "
                                + line);
            }
            String rule = fields.length == 4 && !fields[3].isBlank() ? fields[3].trim() : null;
            return new GeneratedDeck(fields[0], fields[1], CardNameList.parse(fields[2]), rule);
        }

        /**
         * Whether this deck holds exactly {@code cards}, in any order: the
         * same names with the same number of copies each. Content rather than
         * identity, so a deck read from another index instance with the same
         * line still counts as a mirror.
         */
        boolean isMirrorOf(Multiset<String> cards) {
            return cards.equals(ImmutableMultiset.copyOf(cardNames));
        }
    }

    private final List<GeneratedDeck> allDecks;
    /** Never holds an empty list: a set is a key only once a deck of it is added. */
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
        for (String line : CorpusFiles.nonBlankLines(file)) {
            decks.add(GeneratedDeck.parse(line));
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
     * permutation of {@code excludeCards}, or empty if no matching deck
     * exists. The exclusion uses content equality (multiset
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
    public Optional<GeneratedDeck> randomDeckFromSet(
            String setCode, List<String> excludeCards, Random random) {
        Multiset<String> excluded = ImmutableMultiset.copyOf(excludeCards);
        // Filter to non-mirror candidates first, then sample uniformly. This
        // keeps the sample distribution flat when there are many mirrors,
        // unlike rejection sampling.
        List<GeneratedDeck> nonMirror = new ArrayList<>();
        for (GeneratedDeck candidate : decksBySet.getOrDefault(setCode, List.of())) {
            if (!candidate.isMirrorOf(excluded)) {
                nonMirror.add(candidate);
            }
        }
        return anyOf(nonMirror, random);
    }

    /**
     * Pick any random deck from the given set, without excluding mirrors of
     * anything, or empty if the set has no candidates at all.
     *
     * <p>Exists for {@link MatchGenerator#pickDeckB}'s decks-only
     * fallback: a decks-only round has no Forge-legal deck-building path (its
     * set code can be a sentinel like {@code COVERAGE} that resolves against
     * no real Forge edition), so when {@link #randomDeckFromSet} finds no
     * non-mirror candidate, this is the only source left that does not reach
     * Forge. A mirror match still puts every card in the round's decks into
     * a game, which is all a coverage or variant round requires.
     */
    public Optional<GeneratedDeck> randomAnyDeckFromSet(String setCode, Random random) {
        return anyOf(decksBySet.getOrDefault(setCode, List.of()), random);
    }

    /** A uniformly drawn element of {@code decks}, or empty for none. */
    private static Optional<GeneratedDeck> anyOf(List<GeneratedDeck> decks, Random random) {
        return decks.isEmpty()
                ? Optional.empty()
                : Optional.of(decks.get(random.nextInt(decks.size())));
    }
}
