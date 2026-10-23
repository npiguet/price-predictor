package com.pricepredictor.connector;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/** The envelope values, decks and rows the match-corpus tests build from. */
final class MatchFixtures {

    static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    /** The header every fixture row carries unless a test names its own. */
    static final MatchHeader HEADER = header("gen-2");

    private MatchFixtures() {
    }

    /** {@link #HEADER} with its own deck-B method, for rows a test tells apart. */
    static MatchHeader header(String methodB) {
        return new MatchHeader(TS, RUN_ID, "RVR", "forge-best", methodB);
    }

    /** Forty distinct names, {@code <prefix>Card0} to {@code <prefix>Card39}. */
    static List<String> deck40(String prefix) {
        List<String> cards = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cards.add(prefix + "Card" + i);
        }
        return cards;
    }

    /**
     * Per-game results spelled as a row writes them: {@code winners} and
     * {@code starters} are {@code games} / {@code play} sequences such as
     * {@code "ABA"}, read pairwise.
     */
    static List<GameResult> games(String winners, String starters) {
        if (winners.length() != starters.length()) {
            throw new IllegalArgumentException(winners + " / " + starters);
        }
        List<GameResult> games = new ArrayList<>();
        for (int i = 0; i < winners.length(); i++) {
            games.add(new GameResult(side(winners.charAt(i)), side(starters.charAt(i))));
        }
        return games;
    }

    private static Side side(char c) {
        return Side.valueOf(String.valueOf(c));
    }

    /** A {@link #HEADER} match row lasting 47 seconds. */
    static MatchResult matchResult(
            List<String> deckA, List<String> deckB, String winners, String starters) {
        return new MatchResult(HEADER, deckA, deckB, games(winners, starters), 47);
    }

    /** A two-game {@link #HEADER} match row between two {@link #deck40} decks. */
    static MatchResult matchResult() {
        return matchResult(deck40("A"), deck40("B"), "AA", "BA");
    }

    /** A cards-played row under {@code header}: one card played per side, A winning, B starting. */
    static CardsPlayedRow cardsPlayedRow(MatchHeader header) {
        return new CardsPlayedRow(
                header, List.of("X"), List.of("Y"), List.of(), List.of(), Side.A, Side.B);
    }

    /** {@link #cardsPlayedRow(MatchHeader)} under {@link #HEADER}. */
    static CardsPlayedRow cardsPlayedRow() {
        return cardsPlayedRow(HEADER);
    }
}
