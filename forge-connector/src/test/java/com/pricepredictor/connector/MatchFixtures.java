package com.pricepredictor.connector;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/** The envelope values and decks the match-corpus tests build their rows from. */
final class MatchFixtures {

    static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    private MatchFixtures() {
    }

    /** Forty distinct names, {@code <prefix>Card0} to {@code <prefix>Card39}. */
    static List<String> deck40(String prefix) {
        List<String> cards = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cards.add(prefix + "Card" + i);
        }
        return cards;
    }
}
