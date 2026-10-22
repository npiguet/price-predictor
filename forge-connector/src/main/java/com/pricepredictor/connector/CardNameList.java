package com.pricepredictor.connector;

import java.util.Arrays;
import java.util.List;

/**
 * The pipe-separated card-name list every corpus file and worker protocol
 * carries: decks and pools in {@code match-outcomes.txt}, {@code cards-played.txt},
 * {@code pools.txt}, {@code generated-decks.txt}, the seat table, validation
 * matches and {@code DeckBuilderMain}'s stdin/stdout.
 *
 * <p>An empty field is an empty list, not a list holding one empty name.
 * {@code cards-played.txt} writes an empty field whenever a side played
 * nothing, so this is the only reading under which {@link #format} and
 * {@link #parse} round-trip.
 */
final class CardNameList {

    static final String SEPARATOR = "|";

    private CardNameList() {
    }

    /** One field's card names, in order, duplicates kept. */
    static List<String> parse(String field) {
        if (field.isEmpty()) {
            return List.of();
        }
        return Arrays.asList(field.split("\\|", -1));
    }

    /** The field for {@code names}; the empty string for none. */
    static String format(List<String> names) {
        return String.join(SEPARATOR, names);
    }
}
