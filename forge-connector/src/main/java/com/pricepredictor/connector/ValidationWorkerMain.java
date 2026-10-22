package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Entry point for evaluation workers.
 *
 * <p>Initializes the Forge environment, reads a validation matches file where both
 * deck A and deck B are pre-built (40 card names each), plays best-of-K matches,
 * and writes outcomes.
 *
 * <p>Usage:
 * <pre>
 *   java -Dmatches.file=./work/validation-matches-0.txt -Dbest.of=3 -Xmx1200m -cp &lt;classpath&gt;
 *        com.pricepredictor.connector.ValidationWorkerMain
 * </pre>
 */
public class ValidationWorkerMain {

    /**
     * Exit code when the matches or outcomes file cannot be read or written.
     * The supervisor checks only for non-zero, so it need not differ from
     * {@link WorkerProperties#BAD_CONFIG_EXIT}.
     */
    static final int IO_FAILURE_EXIT = 2;

    public static void main(String[] args) {
        String matchesFileProp = WorkerProperties.require("matches.file");
        int bestOf = WorkerProperties.bestOf("best.of", 3);

        Path matchesFile = Path.of(matchesFileProp);
        Path outcomesFile = Path.of(matchesFileProp + "-outcomes.txt");

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Processing: " + matchesFile);

        ValidationMatchPlayer player = new ValidationMatchPlayer(new GamePlayer(bestOf));

        try {
            player.processAll(matchesFile, outcomesFile);
            System.out.println("Validation complete: " + matchesFile);
        } catch (IOException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            throw ToolMain.exit(IO_FAILURE_EXIT);
        }
    }
}
