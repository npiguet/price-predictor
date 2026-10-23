package com.pricepredictor.connector;

import forge.deck.Deck;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
        List<String> lines = Files.readAllLines(matchesFile);
        long completed = countCompletedMatches(outcomesFile);

        System.out.println("Validation: " + lines.size() + " matches, " + completed + " already completed");

        for (int i = (int) completed; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;

            try {
                ParsedMatch parsed = parseLine(line);

                // Both sides are pre-built decks
                Deck deckA = DeckMaterializer.materialize(parsed.deckANames(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);
                Deck deckB = DeckMaterializer.materialize(parsed.deckBNames(),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_DECK);

                GamePlayer.PlayedMatch played = gamePlayer.playMatch(deckA, deckB);
                appendOutcome(outcomesFile, played.winsA(), played.winsB());

                if ((i + 1) % 5 == 0) {
                    System.out.println("Validation: " + (i + 1) + "/" + lines.size() + " matches completed");
                }
            } catch (Exception e) {
                System.err.println("Error on match " + (i + 1) + ": " + e.getMessage());
                // Write a 0;0 outcome to keep line count in sync
                appendOutcome(outcomesFile, 0, 0);
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

    private static void appendOutcome(Path outcomesFile, int winsA, int winsB) throws IOException {
        String line = winsA + ";" + winsB + "\n";
        Files.writeString(outcomesFile, line,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
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
