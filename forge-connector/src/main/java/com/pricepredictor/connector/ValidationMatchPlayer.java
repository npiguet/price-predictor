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
 * <p>For each line: deck A is fully specified (40 card names), pool B is a raw
 * pool from which deck B is built via Forge's {@link forge.gamemodes.limited.SealedDeckBuilder}.
 *
 * <p>Supports crash recovery: on startup, checks the outcomes file line count
 * and skips that many matches in the input file.
 */
public class ValidationMatchPlayer {

    private final DeckBuilder deckBuilder;
    private final GamePlayer gamePlayer;

    public ValidationMatchPlayer(DeckBuilder deckBuilder, GamePlayer gamePlayer) {
        this.deckBuilder = deckBuilder;
        this.gamePlayer = gamePlayer;
    }

    /**
     * Process all matches in the input file, writing outcomes to the derived path.
     *
     * @param matchesFile path to the validation matches file
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

                // Build deck A from card names
                Deck deckA = buildDeckFromNames(parsed.deckANames());

                // Build deck B from pool using Forge's builder
                List<PaperCard> poolB = resolvePaperCards(parsed.poolBNames());
                Deck deckB = deckBuilder.buildDeck(poolB);

                // Play best-of-3
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

    private List<PaperCard> resolvePaperCards(List<String> names) {
        return names.stream()
                .map(name -> FModel.getMagicDb().getCommonCards().getCard(name))
                .filter(card -> card != null)
                .toList();
    }

    // ── Static helpers (package-visible for testing) ────────────────

    record ParsedMatch(List<String> deckANames, List<String> poolBNames) {}

    static ParsedMatch parseLine(String line) {
        String[] parts = line.split(";", 2);
        List<String> deckA = Arrays.asList(parts[0].split("\\|", -1));
        List<String> poolB = Arrays.asList(parts[1].split("\\|", -1));
        return new ParsedMatch(deckA, poolB);
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
