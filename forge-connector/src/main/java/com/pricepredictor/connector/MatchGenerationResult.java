package com.pricepredictor.connector;

import java.util.List;

/**
 * Bundles the parent {@link MatchResult} (one per match) with the per-game
 * {@link CardsPlayedRow} list (one per played game). Returned by
 * {@link MatchGenerator#generateMatch()}.
 *
 * <p>The {@code cardsPlayedRows} list has the same length as the parent's
 * {@code games} string — one entry per played game, in game order. The two
 * pieces are written to different files by the match worker
 * ({@code match-outcomes.txt} via {@link MatchResultWriter},
 * {@code cards-played.txt} via {@link CardsPlayedWriter}).
 */
public record MatchGenerationResult(MatchResult matchResult, List<CardsPlayedRow> cardsPlayedRows) {

    public MatchGenerationResult {
        if (matchResult == null) {
            throw new IllegalArgumentException("matchResult must not be null");
        }
        if (cardsPlayedRows == null) {
            throw new IllegalArgumentException("cardsPlayedRows must not be null");
        }
        if (cardsPlayedRows.size() != matchResult.games().length()) {
            throw new IllegalArgumentException(
                    "cardsPlayedRows size (" + cardsPlayedRows.size()
                            + ") must match parent games length ("
                            + matchResult.games().length() + ")");
        }
    }
}
