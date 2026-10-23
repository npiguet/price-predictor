package com.pricepredictor.connector;

import forge.deck.Deck;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

/**
 * Reads a validation matches file, plays each match, and writes outcomes.
 *
 * <p>Each input line is a {@link ValidationMatch}, both decks fully specified;
 * each writes one {@link ValidationOutcome}, in input order.
 *
 * <p>Supports crash recovery: on startup, checks the outcomes file line count
 * and skips that many matches in the input file.
 */
public class ValidationMatchPlayer {

    /** A progress line every this many matches. */
    static final int PROGRESS_EVERY = 5;

    private final GamePlayer gamePlayer;

    public ValidationMatchPlayer(GamePlayer gamePlayer) {
        this.gamePlayer = gamePlayer;
    }

    /**
     * Process all matches in the input file, writing outcomes to the derived path.
     *
     * @param matchesFile  path to the validation matches file
     * @param outcomesFile path to the outcomes output file
     */
    public void processAll(Path matchesFile, Path outcomesFile) throws IOException {
        // Blank lines are no matches, and the resume count below counts only
        // outcome lines, so the two are indexed over the same non-blank list:
        // a blank line skipped in one and counted in the other would shift
        // every later outcome onto the wrong match.
        List<String> lines = CorpusFiles.nonBlankLines(matchesFile);
        long completed = countCompletedMatches(outcomesFile);
        CorpusWriter<ValidationOutcome> outcomes = new CorpusWriter<>(outcomesFile);

        System.out.println("Validation: " + lines.size() + " matches, " + completed + " already completed");

        for (int i = (int) completed; i < lines.size(); i++) {
            try {
                ValidationMatch match = ValidationMatch.parse(lines.get(i));

                Deck deckA = DeckMaterializer.materialize(match.deckA(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);
                Deck deckB = DeckMaterializer.materialize(match.deckB(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);

                outcomes.write(ValidationOutcome.of(gamePlayer.playMatch(deckA, deckB)));

                if ((i + 1) % PROGRESS_EVERY == 0) {
                    System.out.println("Validation: " + (i + 1) + "/" + lines.size() + " matches completed");
                }
            } catch (Exception e) {
                System.err.println("Error on match " + (i + 1) + ": " + e.getMessage());
                // The filler keeps the outcomes paired with the matches by position.
                outcomes.write(ValidationOutcome.NONE);
            }
        }
    }

    /**
     * Why a validation deck with a name Forge cannot resolve is refused. The
     * evaluation compares two builders' decks; a short one would score the
     * builder on a deck it never built. Refusing lands in the catch above,
     * whose {@link ValidationOutcome#NONE} filler counts no games for either side.
     */
    static final String UNRESOLVED_DECK =
            "A validation match scores the deck a builder chose, so playing it short "
                    + "would score a deck that builder never built.";

    /**
     * An unreadable outcomes file must not read as "nothing completed": the worker would
     * replay every match and append duplicate rows, breaking the positional pairing.
     */
    private static long countCompletedMatches(Path outcomesFile) throws IOException {
        if (!Files.exists(outcomesFile)) return 0;
        try (Stream<String> lines = Files.lines(outcomesFile)) {
            return lines.filter(l -> !l.isBlank()).count();
        }
    }
}
