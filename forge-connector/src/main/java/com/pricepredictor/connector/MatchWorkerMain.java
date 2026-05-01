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
 * <p>Required system properties:
 * <ul>
 *   <li>{@code -Doutput.file=<path>} — where to append match-outcome lines.</li>
 *   <li>{@code -Dmatch.run.id=<UUID>} — supervisor-generated run ID written on every match.</li>
 * </ul>
 *
 * <p>Optional system properties:
 * <ul>
 *   <li>{@code -Dgenerated.decks.file=<path>} — enables self-play mode. The method tag
 *       for each scorer-built deck is read from the deck file's first column
 *       ({@code LABEL} field set by {@code build-decks --label}).</li>
 *   <li>{@code -Dmatch.best.of=<N>} — number of games per match (default 7). Must be a
 *       positive odd integer; any size is valid (Bo1, Bo3, Bo7, Bo17, …).</li>
 * </ul>
 *
 * <p>The worker is terminated externally by the Python supervisor (process.terminate()).
 */
public class MatchWorkerMain {

    static final int DEFAULT_BEST_OF = 7;

    public static void main(String[] args) {
        String outputFileProp = System.getProperty("output.file");
        if (outputFileProp == null) {
            System.err.println("Error: -Doutput.file system property is required");
            System.exit(2);
        }

        String runId = System.getProperty("match.run.id");
        if (runId == null || runId.isBlank()) {
            System.err.println("Error: -Dmatch.run.id system property is required");
            System.exit(2);
        }

        int bestOf = DEFAULT_BEST_OF;
        String bestOfProp = System.getProperty("match.best.of");
        if (bestOfProp != null) {
            try {
                bestOf = Integer.parseInt(bestOfProp);
            } catch (NumberFormatException e) {
                System.err.println(
                        "Error: -Dmatch.best.of must be an integer, got: " + bestOfProp);
                System.exit(2);
            }
            if (bestOf < 1 || bestOf % 2 == 0) {
                System.err.println(
                        "Error: -Dmatch.best.of must be a positive odd integer, got: " + bestOf);
                System.exit(2);
            }
        }

        String generatedDecksProp = System.getProperty("generated.decks.file");

        Path outputFile = Path.of(outputFileProp);

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Starting match generation.");
        System.out.println("Match format: best-of-" + bestOf);

        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        MatchResultWriter writer = new MatchResultWriter(outputFile);

        MatchSource source;
        if (generatedDecksProp == null) {
            MatchGenerator generator = new MatchGenerator(
                    eligibleSets, new DeckBuilder(), new GamePlayer(bestOf), runId);
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
                    new GamePlayer(bestOf), eligibleSets, runId);
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
