package com.pricepredictor.connector;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Per-game card-play record written to {@code output/sealed/cards-played.txt}
 * by {@link CardsPlayedWriter}.
 *
 * <p>Schema (eleven semicolon-separated fields, no trailing {@code ;}):
 * {@code timestamp;run_id;set_code;method_A;method_B;cards_played_A;
 * cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter}
 *
 * <p>Card list columns are pipe-separated and may be empty (encoded as the
 * empty string between two {@code ;}). Multiplicities are preserved.
 *
 * @param timestamp        When the parent match was recorded (UTC). Same value across all games of one match.
 * @param runId            UUID identifying the supervisor invocation that produced the parent match.
 * @param setCode          MTG set code both decks were drawn from.
 * @param methodA          Deck A build method (forge-best/forge-3sub/forge-8sub/random or a LABEL).
 * @param methodB          Deck B build method.
 * @param cardsPlayedA     Non-basic cards in deck A whose name entered the battlefield or stack at least once.
 * @param cardsPlayedB     Same for deck B.
 * @param cardsNotPlayedA  Deck A cards (minus basics) that were not played in this game.
 * @param cardsNotPlayedB  Same for deck B.
 * @param winner           {@code 'A'} or {@code 'B'}; matches the corresponding char of the parent's {@code games}.
 * @param starter          {@code 'A'} or {@code 'B'}; matches the corresponding char of the parent's {@code play}.
 */
public record CardsPlayedRow(
        Instant timestamp,
        String runId,
        String setCode,
        String methodA,
        String methodB,
        List<String> cardsPlayedA,
        List<String> cardsPlayedB,
        List<String> cardsNotPlayedA,
        List<String> cardsNotPlayedB,
        char winner,
        char starter
) {

    public CardsPlayedRow {
        if (winner != 'A' && winner != 'B') {
            throw new IllegalArgumentException(
                    "winner must be 'A' or 'B', got: " + winner);
        }
        if (starter != 'A' && starter != 'B') {
            throw new IllegalArgumentException(
                    "starter must be 'A' or 'B', got: " + starter);
        }
    }

    /** Format this row as one line of {@code cards-played.txt}. */
    public String toLine() {
        return DateTimeFormatter.ISO_INSTANT.format(timestamp)
                + ";" + runId
                + ";" + setCode
                + ";" + methodA
                + ";" + methodB
                + ";" + encode(cardsPlayedA)
                + ";" + encode(cardsPlayedB)
                + ";" + encode(cardsNotPlayedA)
                + ";" + encode(cardsNotPlayedB)
                + ";" + winner
                + ";" + starter;
    }

    private static String encode(List<String> names) {
        return String.join("|", names);
    }
}
