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
        ToolMain.guard(() -> work(args));
    }

    /** The worker itself; an exit it requests ends the process through {@link ToolMain#guard}. */
    static void work(String[] args) {
        String matchesFileProp = WorkerProperties.require("matches.file");
        int bestOf = NumericProperties.bestOf(NumericProperties.IntSetting.bestOf("best.of", 3));

        Path matchesFile = Path.of(matchesFileProp);
        Path outcomesFile = Path.of(matchesFileProp + "-outcomes.txt");

        ForgeEnvironmentInitializer.initializeWorker(System.out, "Processing: " + matchesFile);

        ValidationMatchPlayer player = new ValidationMatchPlayer(new GamePlayer(bestOf));

        try {
            player.processAll(matchesFile, outcomesFile);
            System.out.println("Validation complete: " + matchesFile);
        } catch (IOException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            throw ToolMain.exit(ExitCode.IO_FAILURE);
        }
    }
}
