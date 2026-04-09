package com.pricepredictor.connector;

import java.nio.file.Path;
import java.util.List;

/**
 * Match worker entry point.
 *
 * <p>Initializes the Forge environment, then loops indefinitely generating sealed
 * match outcomes and appending them to the shared output file.
 *
 * <p>Usage (via Python supervisor):
 * <pre>
 *   java -Doutput.file=./output/sealed/match-outcomes.txt -Xmx1200m -cp &lt;classpath&gt;
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

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Starting match generation.");

        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        MatchGenerator generator = new MatchGenerator(eligibleSets, new DeckBuilder(), new GamePlayer());
        MatchResultWriter writer = new MatchResultWriter(outputFile);

        long count = 0;
        while (true) {
            try {
                MatchResult result = generator.generateMatch();
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
}
