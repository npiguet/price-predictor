package com.pricepredictor.connector;

import java.nio.file.Path;
import java.util.List;

/**
 * Match worker entry point.
 *
 * <p>Initializes the Forge environment, then loops indefinitely generating sealed
 * match outcomes and appending them to the shared output file.
 *
 * <p>Two modes, selected by the optional {@code -Dgenerated.decks.file} system property:
 * <ul>
 *   <li><b>Phase 0 (default)</b>: When the property is absent, each match uses
 *       {@link MatchGenerator}: pick a random eligible set, generate two pools,
 *       build two decks, play.</li>
 *   <li><b>Self-play</b>: When the property points to a generated-decks file,
 *       deck A is sampled from that file and deck B is built by
 *       {@link SelfPlayMatchGenerator}.</li>
 * </ul>
 *
 * <p>Usage (via Python supervisor):
 * <pre>
 *   java -Doutput.file=./output/sealed/match-outcomes.txt
 *        [-Dgenerated.decks.file=./output/sealed/generated-decks.txt]
 *        -Xmx1200m -cp &lt;classpath&gt;
 *        com.pricepredictor.connector.MatchWorkerMain
 * </pre>
 *
 * <p>The worker is terminated externally by the Python supervisor (process.terminate()).
 */
public class MatchWorkerMain {

    public static void main(String[] args) {
        String outputFileProp = System.getProperty("output.file");
        if (outputFileProp == null) {
            System.err.println("Error: -Doutput.file system property is required");
            System.exit(2);
        }

        Path outputFile = Path.of(outputFileProp);
        String generatedDecksProp = System.getProperty("generated.decks.file");

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Starting match generation.");

        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        MatchResultWriter writer = new MatchResultWriter(outputFile);

        MatchSource source;
        if (generatedDecksProp == null) {
            MatchGenerator generator = new MatchGenerator(
                    eligibleSets, new DeckBuilder(), new GamePlayer());
            source = generator::generateMatch;
        } else {
            Path generatedDecks = Path.of(generatedDecksProp);
            System.out.println("Self-play mode: loading generated decks from " + generatedDecks);
            GeneratedDecksIndex index;
            try {
                index = GeneratedDecksIndex.load(generatedDecks);
            } catch (Exception e) {
                System.err.println("Error loading generated-decks file: " + e.getMessage());
                System.exit(2);
                return;
            }
            System.out.println("Loaded " + index.size() + " generated decks.");
            SelfPlayMatchGenerator selfPlay = new SelfPlayMatchGenerator(
                    index, new DeckBuilder(), new PoolGenerator(),
                    new GamePlayer(), eligibleSets);
            source = selfPlay::generateMatch;
        }

        long count = 0;
        while (true) {
            try {
                MatchResult result = source.next();
                writer.write(result);
                count++;
                if (count % 10 == 0) {
                    System.out.println("Worker: " + count + " matches generated");
                    System.out.flush();
                }
            } catch (Exception e) {
                System.err.println("Error generating match: " + e.getMessage());
                e.printStackTrace(System.err);
                // Continue on non-fatal errors; fatal errors (OOM, etc.) will propagate
            }
        }
    }

    @FunctionalInterface
    private interface MatchSource {
        MatchResult next();
    }
}
