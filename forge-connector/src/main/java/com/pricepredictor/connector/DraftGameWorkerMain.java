package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Entry point for draft game-evaluation workers.
 *
 * <p>Loads a seat table written by {@code python -m draft play-draft-games}, then loops:
 * draw a pod, draw two distinct seats of it, play a best-of-K match, append one
 * match-outcome row. The worker never terminates on its own — the supervisor kills it,
 * either because the run is over or because it is being recycled.
 *
 * <p>Unlike {@link ValidationWorkerMain} this worker is autonomous: it is handed a
 * population rather than a work list, and it chooses its own pairings. Nothing is
 * positionally aligned, so a pairing whose match fails writes no row at all.
 *
 * <p>Usage:
 * <pre>
 *   java -Dseats.file=./work/seats.txt -Doutput.file=./output/draft/draft-games.txt
 *        -Drun.id=&lt;uuid&gt; -Dbest.of=3 -Dinclude.mirrors=false -Xmx1200m
 *        -cp &lt;classpath&gt; com.pricepredictor.connector.DraftGameWorkerMain
 * </pre>
 */
public class DraftGameWorkerMain {

    public static void main(String[] args) {
        String seatsFileProp = requiredProperty("seats.file");
        String outputFileProp = requiredProperty("output.file");
        String runId = requiredProperty("run.id");

        int bestOf = 3;
        try {
            bestOf = Integer.parseInt(System.getProperty("best.of", "3"));
        } catch (NumberFormatException e) {
            System.err.println("Warning: invalid -Dbest.of value, using 3");
        }

        boolean includeMirrors = Boolean.parseBoolean(
                System.getProperty("include.mirrors", "false"));

        Path seatsFile = Path.of(seatsFileProp);
        Path outputFile = Path.of(outputFileProp);

        System.out.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Forge initialized. Seat table: " + seatsFile);

        run(seatsFile, outputFile, runId, bestOf, includeMirrors);
    }

    /**
     * Load the seat table and play pairings until the process is killed.
     *
     * <p>A pairing that throws is abandoned without writing a row and the loop
     * draws another. Nothing was reserved for it, so nothing is lost but time.
     */
    static void run(
            Path seatsFile,
            Path outputFile,
            String runId,
            int bestOf,
            boolean includeMirrors) {
        SeatTableIndex index;
        try {
            index = SeatTableIndex.load(seatsFile, includeMirrors);
        } catch (IOException e) {
            System.err.println("Fatal error reading " + seatsFile + ": " + e.getMessage());
            System.exit(2);
            return;
        }

        System.out.println(
                "Seat table: " + index.seatCount() + " seats, "
                        + index.eligiblePodCount() + " pods able to yield a pairing");

        if (index.eligiblePodCount() == 0) {
            System.err.println("Error: no pod in the seat table can yield a pairing");
            System.exit(2);
            return;
        }

        DraftGamePlayer player = new DraftGamePlayer(
                new GamePlayer(bestOf), new MatchResultWriter(outputFile), runId);

        long played = 0;
        while (true) {
            try {
                SeatTableIndex.Pairing pairing = index.randomPairing(includeMirrors);
                if (pairing == null) {
                    System.err.println("Error: no pairing available; stopping");
                    return;
                }
                player.play(pairing);
                played++;
                if (played % 5 == 0) {
                    System.out.println("Worker: " + played + " matches played");
                }
            } catch (Exception e) {
                // No row is written. The supervisor sees this only as a slower
                // match rate; it never reserved this pairing for us.
                System.err.println("Error playing pairing: " + e.getMessage());
            }
        }
    }

    /** Read a required system property, or exit 2 naming the one that is missing. */
    private static String requiredProperty(String name) {
        String value = System.getProperty(name);
        if (value == null || value.isBlank()) {
            System.err.println("Error: -D" + name + " system property is required");
            System.exit(2);
        }
        return value;
    }
}
