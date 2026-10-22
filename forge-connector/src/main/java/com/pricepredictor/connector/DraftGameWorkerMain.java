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
        String seatsFileProp = WorkerProperties.require("seats.file");
        String outputFileProp = WorkerProperties.require("output.file");
        String runId = WorkerProperties.require("run.id");
        int bestOf = WorkerProperties.bestOf("best.of", 3);
        boolean includeMirrors = WorkerProperties.flag("include.mirrors");

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
     * draws another. Nothing was reserved for it, so nothing is lost but time —
     * unless every pairing throws, which {@link WorkerLoop} turns into exit 1.
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
            throw WorkerProperties.badConfig("reading " + seatsFile + ": " + e.getMessage());
        }

        System.out.println(
                "Seat table: " + index.seatCount() + " seats, "
                        + index.eligiblePodCount() + " pods able to yield a pairing");

        if (index.eligiblePodCount() == 0) {
            throw WorkerProperties.badConfig("no pod in the seat table can yield a pairing");
        }

        DraftGamePlayer player = new DraftGamePlayer(
                new GamePlayer(bestOf), new MatchResultWriter(outputFile), runId);

        WorkerLoop.runUntilBroken(
                () -> {
                    // Never empty: a table with no eligible pod exited above.
                    player.play(index.randomPairing().orElseThrow());
                    return WorkerLoop.Outcome.COMPLETED;
                },
                new WorkerLoop.Progress(
                        System.out, 5, "matches played", "Error playing pairing"));
    }
}
