package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.model.FModel;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

/**
 * CLI entry point for batch Forge deck building.
 *
 * <p>Reads sealed pools from stdin (one pool per line, card names pipe-separated)
 * and writes the built 40-card deck to stdout for each pool (one deck per line,
 * card names pipe-separated). Uses Forge's {@link forge.gamemodes.limited.SealedDeckBuilder}
 * directly with no random variants — this is the pure Forge baseline used in evaluation.
 *
 * <p>A single JVM invocation processes all N pools, avoiding repeated Forge
 * initialization overhead.
 *
 * <p>Usage:
 * <pre>
 *   echo "CardA|CardB|..." | java -cp &lt;classpath&gt;
 *       com.pricepredictor.connector.DeckBuilderMain
 * </pre>
 *
 * <p>Exits 0 on success, 1 on error.
 */
public class DeckBuilderMain {

    public static void main(String[] args) {
        // Capture real stdout, then redirect System.out to stderr so Forge's
        // logging (e.g. "Read cards: ...", "Basics[Plains]: ...") doesn't
        // contaminate the deck output stream consumed by the Python connector.
        PrintStream realOut = System.out;
        System.setOut(System.err);

        ForgeEnvironmentInitializer.initialize();

        DeckBuilder deckBuilder = new DeckBuilder();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                List<String> cardNames = List.of(line.split("\\|", -1));
                List<PaperCard> pool = resolveCards(cardNames);
                Deck deck = deckBuilder.buildStandard(pool);
                realOut.println(deckToLine(deck));
                realOut.flush();
            }

            System.exit(0);

        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static List<PaperCard> resolveCards(List<String> names) {
        List<PaperCard> cards = new ArrayList<>();
        for (String name : names) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            if (card != null) {
                cards.add(card);
            }
        }
        return cards;
    }

    static String deckToLine(Deck deck) {
        StringJoiner joiner = new StringJoiner("|");
        for (PaperCard card : deck.get(DeckSection.Main).toFlatList()) {
            joiner.add(card.getName());
        }
        return joiner.toString();
    }
}
