package com.pricepredictor.connector.effects;

import java.util.List;
import java.util.Optional;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * The rules a game's two decks are aimed at (FR-058d), each null for a deck
 * aimed at none.
 *
 * <p>A steered variant round writes each deck's rule in a fourth column of
 * the decks file. A game is steered when either of its decks carries one; no
 * cap property says so, because the worker plays steered and unsteered decks
 * alike and only the deck it drew knows.
 *
 * @param deckA the rule deck A is aimed at, or null
 * @param deckB the rule deck B is aimed at, or null
 */
public record DeckRules(String deckA, String deckB) {

    /** A game whose decks are aimed at no rule. */
    public static final DeckRules NONE = new DeckRules(null, null);

    public static DeckRules of(String deckA, String deckB) {
        return deckA == null && deckB == null ? NONE : new DeckRules(deckA, deckB);
    }

    /** Whether either deck carries a rule. */
    public boolean steered() {
        return deckA != null || deckB != null;
    }

    /**
     * What every record of the game carries as {@code aimed_at}: in a steered
     * game both decks' rules, de-duplicated and sorted, in place of the run's
     * list; otherwise the run's list, or null for a run aimed at nothing.
     * A record already aimed (a directed fork's) keeps its own whatever this
     * says ({@link CollectorContext#emit}).
     */
    public List<String> aimedAt(Optional<List<String>> runAimedAt) {
        if (!steered()) {
            return runAimedAt.orElse(null);
        }
        TreeSet<String> rules = new TreeSet<>();
        Stream.of(deckA, deckB).filter(rule -> rule != null).forEach(rules::add);
        return List.copyOf(rules);
    }
}
