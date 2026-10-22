package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.item.PaperCard;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

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

                List<PaperCard> pool = DeckMaterializer.resolve(CardNameList.parse(line),
                        DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_POOL);
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

    /**
     * Why a pool with a name Forge cannot resolve fails the whole call. This
     * deck is the Forge baseline the scorer's deck for the same pool is
     * compared against; built from fewer cards, the comparison would be
     * between two different pools. The Python side reads one deck per pool
     * line, so there is no per-line way to skip one: exit 1 and say why.
     */
    static final String UNRESOLVED_POOL =
            "The Forge baseline must be built from the very pool the scorer saw, "
                    + "and a pool missing cards is a different pool.";

    static String deckToLine(Deck deck) {
        return CardNameList.format(DeckMaterializer.names(deck));
    }
}
