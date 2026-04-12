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

    public static void main(String[] args) {
        String matchesFileProp = System.getProperty("matches.file");
        if (matchesFileProp == null) {
            System.err.println("Error: -Dmatches.file system property is required");
            System.exit(2);
        }

        int bestOf = 3;
        try {
            bestOf = Integer.parseInt(System.getProperty("best.of", "3"));
        } catch (NumberFormatException e) {
            System.err.println("Warning: invalid -Dbest.of value, using 3");
        }

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
            System.exit(2);
        }
    }
}
