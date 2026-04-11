package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Entry point for evaluation workers.
 *
 * <p>Initializes the Forge environment, reads a validation matches file,
 * builds deck B from each pool using Forge's SealedDeckBuilder,
 * plays best-of-3 matches, and writes outcomes.
 *
 * <p>Usage:
 * <pre>
 *   java -Dmatches.file=./work/validation-matches-0.txt -Xmx1200m -cp &lt;classpath&gt;
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

        Path matchesFile = Path.of(matchesFileProp);
        Path outcomesFile = Path.of(matchesFileProp + "-outcomes.txt");

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Processing: " + matchesFile);

        ValidationMatchPlayer player = new ValidationMatchPlayer(
                new DeckBuilder(), new GamePlayer()
        );

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
