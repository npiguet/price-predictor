package com.pricepredictor.connector;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * Match worker entry point.
 *
 * <p>Initializes the Forge environment, then loops indefinitely generating sealed
 * match outcomes via {@link MatchGenerator} and appending them to the shared
 * output file.
 *
 * <p>Required system properties:
 * <ul>
 *   <li>{@code -Doutput.file=<path>} — where to append match-outcome lines.</li>
 *   <li>{@code -Dmatch.run.id=<UUID>} — supervisor-generated run ID written on every match.</li>
 * </ul>
 *
 * <p>Optional system properties (configure side-A / side-B sources):
 * <ul>
 *   <li>{@code -Dside.a.decks.file=<path>} — sample deck A from this generated-decks
 *       file instead of using the 4 Forge methods on side A. The method tag for
 *       each scorer-built deck is read from the deck file's first column
 *       ({@code LABEL} field set by {@code build-decks --label}).</li>
 *   <li>{@code -Dside.b.decks.file=<path>} — additionally allow deck B to be
 *       sampled from this generated-decks file (the 4 Forge methods are still
 *       used). Weight controlled by {@code -Dside.b.decks.weight}.</li>
 *   <li>{@code -Dside.b.decks.weight=<int>} — weight of the file-sampled side-B
 *       method relative to the 4 Forge methods (which carry total weight 10).
 *       Default 4. Only meaningful when {@code -Dside.b.decks.file} is also set.</li>
 *   <li>{@code -Dmatch.best.of=<N>} — number of games per match (default 7). Must be a
 *       positive odd integer; any size is valid (Bo1, Bo3, Bo7, Bo17, …).</li>
 * </ul>
 *
 * <p>The worker is terminated externally by the Python supervisor (process.terminate()).
 */
public class MatchWorkerMain {

    static final int DEFAULT_BEST_OF = 7;
    static final int DEFAULT_SIDE_B_WEIGHT = 4;

    /**
     * Validated worker configuration parsed from the JVM system properties.
     * A pure value object — no Forge or filesystem touching here so the parse
     * step can be exercised independently of the rest of the worker.
     */
    record WorkerConfig(
            Path outputFile,
            String runId,
            int bestOf,
            String sideADecksFile,
            String sideBDecksFile,
            int sideBWeight) {}

    public static void main(String[] args) {
        WorkerConfig config = parseConfig();
        initializeForge(config);
        runForever(config);
    }

    /**
     * Read every JVM system property the worker depends on, validate it, and
     * package the result as a {@link WorkerConfig}. Exits the process with
     * code {@code 2} on any malformed input.
     */
    static WorkerConfig parseConfig() {
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

        int bestOf = parseIntProp(
                "match.best.of", DEFAULT_BEST_OF,
                v -> v >= 1 && v % 2 == 1, "must be a positive odd integer");

        String sideAProp = System.getProperty("side.a.decks.file");
        String sideBProp = System.getProperty("side.b.decks.file");
        String sideBWeightProp = System.getProperty("side.b.decks.weight");

        if (sideBProp == null && sideBWeightProp != null) {
            System.err.println(
                    "Error: -Dside.b.decks.weight is only valid with -Dside.b.decks.file");
            System.exit(2);
        }

        int sideBWeight = parseIntProp(
                "side.b.decks.weight", DEFAULT_SIDE_B_WEIGHT,
                v -> v >= 1, "must be >= 1");

        return new WorkerConfig(
                Path.of(outputFileProp), runId, bestOf,
                sideAProp, sideBProp, sideBWeight);
    }

    /**
     * Parse an integer JVM system property, validating against {@code isValid}.
     * Returns {@code defaultValue} when the property is unset; calls
     * {@link System#exit} with code {@code 2} on a parse error or constraint
     * violation. Used for both {@code -Dmatch.best.of} and
     * {@code -Dside.b.decks.weight}.
     */
    static int parseIntProp(
            String name, int defaultValue, IntPredicate isValid, String constraintMsg) {
        String prop = System.getProperty(name);
        if (prop == null) {
            return defaultValue;
        }
        int value;
        try {
            value = Integer.parseInt(prop);
        } catch (NumberFormatException e) {
            System.err.println(
                    "Error: -D" + name + " must be an integer, got: " + prop);
            System.exit(2);
            return -1;  // unreachable
        }
        if (!isValid.test(value)) {
            System.err.println(
                    "Error: -D" + name + " " + constraintMsg + ", got: " + value);
            System.exit(2);
        }
        return value;
    }

    private static void initializeForge(WorkerConfig config) {
        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Starting match generation.");
        System.out.println("Match format: best-of-" + config.bestOf());
    }

    private static void runForever(WorkerConfig config) {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        GeneratedDecksIndex sideAIndex = loadIndex(config.sideADecksFile(), "side-A");
        GeneratedDecksIndex sideBIndex = loadIndex(config.sideBDecksFile(), "side-B");

        MatchResultWriter writer = new MatchResultWriter(config.outputFile());
        CardsPlayedWriter cardsPlayedWriter = new CardsPlayedWriter(
                cardsPlayedPath(config.outputFile()));
        MatchGenerator generator = new MatchGenerator(
                eligibleSets, new DeckBuilder(), new GamePlayer(config.bestOf()), config.runId(),
                sideAIndex, sideBIndex, config.sideBWeight());

        long count = 0;
        while (true) {
            try {
                MatchGenerationResult result = generator.generateMatch();
                writer.write(result.matchResult());
                for (CardsPlayedRow row : result.cardsPlayedRows()) {
                    cardsPlayedWriter.write(row);
                }
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

    /**
     * Resolve the cards-played output path next to the match-outcomes file.
     * Both files share the {@code output/sealed/} directory by spec
     * (files.md), so we only need to swap the filename.
     */
    static Path cardsPlayedPath(Path matchOutcomesFile) {
        Path parent = matchOutcomesFile.getParent();
        if (parent == null) {
            return Paths.get("cards-played.txt");
        }
        return parent.resolve("cards-played.txt");
    }

    private static GeneratedDecksIndex loadIndex(String pathProp, String label) {
        if (pathProp == null) {
            return null;
        }
        Path path = Path.of(pathProp);
        System.out.println("Loading " + label + " decks from " + path);
        try {
            GeneratedDecksIndex index = GeneratedDecksIndex.load(path);
            System.out.println("Loaded " + index.size() + " " + label + " decks.");
            return index;
        } catch (Exception e) {
            System.err.println(
                    "Error loading " + label + " decks file: " + e.getMessage());
            System.exit(2);
            return null;  // unreachable
        }
    }
}
