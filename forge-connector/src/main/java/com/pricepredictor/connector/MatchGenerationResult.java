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
 *
 * <p>{@code randomSeat} marks a match one of whose seats was the random seat
 * (FR-021). Such a match is effect records only: the worker writes its
 * progress line and neither sealed row (FR-025), because one seat played at
 * random and the outcome says nothing about either deck. The rows are still
 * carried, built the same way, so the result is one shape whatever the
 * seats; the row-count check is relaxed for it alone, since nothing reads
 * the rows.
 */
public record MatchGenerationResult(
        MatchResult matchResult, List<CardsPlayedRow> cardsPlayedRows, boolean randomSeat) {

    public MatchGenerationResult {
        if (matchResult == null) {
            throw new IllegalArgumentException("matchResult must not be null");
        }
        if (cardsPlayedRows == null) {
            throw new IllegalArgumentException("cardsPlayedRows must not be null");
        }
        boolean rowsMatch = cardsPlayedRows.size() == matchResult.games().length();
        if (!rowsMatch && !(randomSeat && cardsPlayedRows.isEmpty())) {
            throw new IllegalArgumentException(
                    "cardsPlayedRows size (" + cardsPlayedRows.size()
                            + ") must match parent games length ("
                            + matchResult.games().length() + ")");
        }
    }

    /** An ordinary two-AI match, whose rows both writers take. */
    public MatchGenerationResult(MatchResult matchResult, List<CardsPlayedRow> cardsPlayedRows) {
        this(matchResult, cardsPlayedRows, false);
    }
}
