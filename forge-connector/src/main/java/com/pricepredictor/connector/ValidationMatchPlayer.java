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
 * <p>For each line: both deck A and deck B are fully specified as 40 card names.
 * The format is {@code deckA_card1|...|card40;deckB_card1|...|card40}.
 *
 * <p>Supports crash recovery: on startup, checks the outcomes file line count
 * and skips that many matches in the input file.
 */
public class ValidationMatchPlayer {

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
        List<String> lines = Files.readAllLines(matchesFile).stream()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .toList();
        long completed = countCompletedMatches(outcomesFile);
        LineAppender outcomes = new LineAppender(outcomesFile);

        System.out.println("Validation: " + lines.size() + " matches, " + completed + " already completed");

        for (int i = (int) completed; i < lines.size(); i++) {
            try {
                ParsedMatch parsed = parseLine(lines.get(i));

                Deck deckA = DeckMaterializer.materialize(parsed.deckANames(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);
                Deck deckB = DeckMaterializer.materialize(parsed.deckBNames(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);

                GamePlayer.PlayedMatch played = gamePlayer.playMatch(deckA, deckB);
                outcomes.append(played.winsA() + ";" + played.winsB());

                if ((i + 1) % 5 == 0) {
                    System.out.println("Validation: " + (i + 1) + "/" + lines.size() + " matches completed");
                }
            } catch (Exception e) {
                System.err.println("Error on match " + (i + 1) + ": " + e.getMessage());
                // A 0;0 filler keeps the outcomes paired with the matches by position.
                outcomes.append("0;0");
            }
        }
    }

    /**
     * Why a validation deck with a name Forge cannot resolve is refused. The
     * evaluation compares two builders' decks; a short one would score the
     * builder on a deck it never built. Refusing lands in the catch below,
     * whose {@code 0;0} filler counts no games for either side.
     */
    static final String UNRESOLVED_DECK =
            "A validation match scores the deck a builder chose, so playing it short "
                    + "would score a deck that builder never built.";

    private record ParsedMatch(List<String> deckANames, List<String> deckBNames) {}

    private static ParsedMatch parseLine(String line) {
        String[] parts = line.split(";", 2);
        return new ParsedMatch(CardNameList.parse(parts[0]), CardNameList.parse(parts[1]));
    }

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
