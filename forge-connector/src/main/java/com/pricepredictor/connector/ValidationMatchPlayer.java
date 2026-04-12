package com.pricepredictor.connector;

import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.model.FModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;

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
                Deck deckA = buildDeckFromNames(parsed.deckANames());
                Deck deckB = buildDeckFromNames(parsed.deckBNames());

                int[] result = gamePlayer.playMatch(deckA, deckB);
                appendOutcome(outcomesFile, result[0], result[1]);

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

    private Deck buildDeckFromNames(List<String> cardNames) {
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        for (String name : cardNames) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            if (card != null) {
                main.add(card);
            }
        }
        return deck;
    }

    // ── Static helpers (package-visible for testing) ────────────────

    record ParsedMatch(List<String> deckANames, List<String> deckBNames) {}

    static ParsedMatch parseLine(String line) {
        String[] parts = line.split(";", 2);
        List<String> deckA = Arrays.asList(parts[0].split("\\|", -1));
        List<String> deckB = Arrays.asList(parts[1].split("\\|", -1));
        return new ParsedMatch(deckA, deckB);
    }

    static void appendOutcome(Path outcomesFile, int winsA, int winsB) throws IOException {
        String line = winsA + ";" + winsB + "\n";
        Files.writeString(outcomesFile, line,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    static long countCompletedMatches(Path outcomesFile) {
        if (!Files.exists(outcomesFile)) return 0;
        try {
            return Files.lines(outcomesFile).filter(l -> !l.isBlank()).count();
        } catch (IOException e) {
            return 0;
        }
    }
}
