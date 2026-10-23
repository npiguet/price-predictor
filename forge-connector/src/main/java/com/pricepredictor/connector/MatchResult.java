package com.pricepredictor.connector;

import java.time.Instant;
import java.util.List;
import java.util.function.Function;

/**
 * Value object capturing the outcome of one sealed best-of-N match, including
 * the metadata needed to slice the training corpus later (run ID, set code,
 * deck-build methods, per-game winner/play-first sequences, duration).
 *
 * @param header           The five leading fields this row shares with its cards-played rows.
 * @param deckA            All 40 cards in deck A (including basic lands, duplicates repeat).
 * @param deckB            All 40 cards in deck B.
 * @param games            One entry per game played, in game order: between
 *                         {@code ceil(N/2)} and {@code N} of them for a best-of-N match.
 *                         Written as the row's {@code games} (winners) and {@code play}
 *                         (starters) sequences, e.g. {@code ABB} and {@code BAB}.
 * @param durationSeconds  Wall-clock seconds spent playing the match.
 */
public record MatchResult(
        MatchHeader header,
        List<String> deckA,
        List<String> deckB,
        List<GameResult> games,
        int durationSeconds
) {

    public MatchResult {
        if (games == null || games.isEmpty()) {
            // GamePlayer skips draws and aborted games, so a match can end
            // with none; a row with empty sequences would be blank corpus.
            throw new IllegalArgumentException("games must be non-empty, got: " + games);
        }
        games = List.copyOf(games);
        if (durationSeconds < 0) {
            throw new IllegalArgumentException(
                    "durationSeconds must be non-negative, got " + durationSeconds);
        }
    }

    /**
     * The row for a match just played, timestamped now.
     *
     * <p>The deck names are passed in rather than read off the played decks,
     * because callers spell them differently: the sealed generator has the
     * names it sampled, the draft player the names of the decks it built.
     */
    public static MatchResult of(
            String runId,
            String setCode,
            String methodA,
            String methodB,
            List<String> deckA,
            List<String> deckB,
            GamePlayer.PlayedMatch played) {
        return new MatchResult(
                new MatchHeader(Instant.now(), runId, setCode, methodA, methodB),
                deckA, deckB, played.results(), played.durationSeconds());
    }

    /**
     * Format this result as one line of {@code match-outcomes.txt}: ten
     * semicolon-separated fields,
     * {@code timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s},
     * the two decks as pipe-separated card names.
     */
    public String toLine() {
        return header.line(
                CardNameList.format(deckA), CardNameList.format(deckB),
                sequence(GameResult::winner), sequence(GameResult::starter),
                String.valueOf(durationSeconds));
    }

    /** One side character per game, in game order. */
    private String sequence(Function<GameResult, Side> sideOf) {
        StringBuilder sb = new StringBuilder(games.size());
        for (GameResult game : games) {
            sb.append(sideOf.apply(game));
        }
        return sb.toString();
    }
}
