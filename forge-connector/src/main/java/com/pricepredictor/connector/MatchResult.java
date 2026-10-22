package com.pricepredictor.connector;

import java.time.Instant;
import java.util.List;

/**
 * Value object capturing the outcome of one sealed best-of-N match, including
 * the metadata needed to slice the training corpus later (run ID, set code,
 * deck-build methods, per-game winner/play-first sequences, duration).
 *
 * @param timestamp        When the match finished (UTC).
 * @param runId            UUID identifying the supervisor invocation that produced this match.
 * @param setCode          MTG set code both decks were drawn from.
 * @param methodA          How deck A was built (see {@link DeckBuilder} method tags, or the
 *                         per-deck {@code LABEL} field when deck A came from a generated-decks file).
 * @param methodB          How deck B was built (same enum as {@code methodA}).
 * @param deckA            All 40 cards in deck A (including basic lands, duplicates repeat).
 * @param deckB            All 40 cards in deck B.
 * @param games            Per-game winner sequence, e.g. {@code "ABB"} means A won game 1,
 *                         B won games 2 and 3. Between {@code ceil(N/2)} and {@code N}
 *                         long for a best-of-N match.
 * @param play             Per-game play-first sequence, same length as {@code games}, e.g.
 *                         {@code "BAB"} means B was on the play in games 1 and 3.
 * @param durationSeconds  Wall-clock seconds spent playing the match.
 */
public record MatchResult(
        Instant timestamp,
        String runId,
        String setCode,
        String methodA,
        String methodB,
        List<String> deckA,
        List<String> deckB,
        String games,
        String play,
        int durationSeconds
) {

    public MatchResult {
        if (games == null || games.isEmpty()) {
            throw new IllegalArgumentException(
                    "games must be non-empty, got: " + games);
        }
        if (play == null || play.length() != games.length()) {
            throw new IllegalArgumentException(
                    "play length must match games length: games=" + games + " play=" + play);
        }
        for (int i = 0; i < games.length(); i++) {
            char g = games.charAt(i);
            char p = play.charAt(i);
            if (g != 'A' && g != 'B') {
                throw new IllegalArgumentException(
                        "games must contain only 'A' or 'B', got: " + games);
            }
            if (p != 'A' && p != 'B') {
                throw new IllegalArgumentException(
                        "play must contain only 'A' or 'B', got: " + play);
            }
        }
        if (durationSeconds < 0) {
            throw new IllegalArgumentException(
                    "durationSeconds must be non-negative, got " + durationSeconds);
        }
    }

    /**
     * Format this result as one line of {@code match-outcomes.txt}: ten
     * semicolon-separated fields,
     * {@code timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s},
     * the two decks as pipe-separated card names.
     */
    public String toLine() {
        return CorpusLine.of(timestamp, runId, setCode, methodA, methodB,
                CardNameList.format(deckA), CardNameList.format(deckB),
                games, play, String.valueOf(durationSeconds));
    }
}
