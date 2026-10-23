package com.pricepredictor.connector;

import java.nio.file.Path;

/**
 * The two sealed corpus files a match worker appends to, written together or
 * not at all: {@code match-outcomes.txt} takes one row per match and
 * {@code cards-played.txt}, beside it, one row per game.
 *
 * <p>One value for the pair because one gate governs both: a records-only
 * worker (coverage and variant rounds) has no sealed corpus at all, and a
 * worker that has one writes every match to both files, since
 * {@code cards-played.txt} rows are joined to their parent match by position.
 */
record SealedCorpus(CorpusWriter<MatchResult> outcomes, CorpusWriter<CardsPlayedRow> cardsPlayed) {

    /**
     * The corpus whose match outcomes go to {@code matchOutcomesFile} and
     * whose cards-played rows go to {@code cards-played.txt} in the same
     * directory, where both files live by spec.
     */
    static SealedCorpus beside(Path matchOutcomesFile) {
        return new SealedCorpus(
                new CorpusWriter<>(matchOutcomesFile),
                new CorpusWriter<>(cardsPlayedPath(matchOutcomesFile)));
    }

    /** {@code cards-played.txt} next to the match-outcomes file. */
    static Path cardsPlayedPath(Path matchOutcomesFile) {
        return matchOutcomesFile.resolveSibling("cards-played.txt");
    }

    /** Append the match's outcome row and each of its games' cards-played rows. */
    void record(MatchGenerationResult result) {
        outcomes.write(result.matchResult());
        for (CardsPlayedRow row : result.cardsPlayedRows()) {
            cardsPlayed.write(row);
        }
    }
}
