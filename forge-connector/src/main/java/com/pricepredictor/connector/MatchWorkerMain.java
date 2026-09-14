package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.AttributionMode;
import com.pricepredictor.connector.effects.PatchHooks;
import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.RecordShardWriter;
import forge.util.MyRandom;

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
 *   <li>{@code -Dsealed.decks.only=true} — force side B to always sample the
 *       decks file instead of rolling the 4 Forge methods; deck A already
 *       comes from the file whenever {@code -Dside.a.decks.file} is set. For
 *       decks (coverage, variant) that belong to no set and carry a sentinel
 *       set code the Forge-method branch would fail to resolve against
 *       Forge's booster/edition tables. Default {@code false}. See
 *       {@link MatchGenerator#rollIsFileSample()}.</li>
 *   <li>{@code -Dmatch.best.of=<N>} — number of games per match (default 7). Must be a
 *       positive odd integer; any size is valid (Bo1, Bo3, Bo7, Bo17, …).</li>
 *   <li>{@code -Deffect.worker.lifetime=<token>} — pins the per-JVM token that makes
 *       record and game ids unique across worker restarts. Minted from the clock
 *       when absent, which is the normal case; only a test should set it.</li>
 *   <li>{@code -Deffect.snapshot.tiers=1,2,3} — the inclusion depth every snapshot
 *       in the run is built at, as a prefix of {@code 1,2,3,4}. A run-level
 *       property rather than a per-collector one: a depth that varies by record
 *       kind puts collection metadata into {@code state.tiers}, where a model can
 *       read it. Parsed by {@code PatchedCollectors.CollectionCaps} along with the
 *       rest of the {@code effect.*} caps, which that record documents; echoed at
 *       startup because it is also the largest single lever on shard size.</li>
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
            int sideBWeight,
            Path effectRecordsDir,
            int workerIndex,
            String workerLifetime,
            boolean decksOnly) {

        /**
         * Records-only: no sealed corpus is written at all.
         *
         * <p>The coverage and variant collectors reuse this worker, and they
         * must never touch {@code match-outcomes.txt} or {@code cards-played.txt}
         * — their decks are built for coverage, not for a fair self-play sample,
         * and mixing them into the sealed corpus would corrupt the scorer's
         * training data. The guard lives here rather than in the Python
         * supervisor because it is this class that constructs the writers.
         */
        boolean recordsOnly() {
            return outputFile == null;
        }

        boolean collectsEffectRecords() {
            return effectRecordsDir != null;
        }
    }

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
        String effectRecordsProp = System.getProperty("effect.records.dir");
        if (outputFileProp == null && effectRecordsProp == null) {
            System.err.println(
                    "Error: -Doutput.file is required unless -Deffect.records.dir"
                            + " is set (records-only mode)");
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

        // A coverage or variant deck belongs to no set, so the roll's
        // Forge-method branch would resolve its sentinel set code against
        // Forge's booster/edition tables and throw. This forces the roll to
        // the decks file every time. See MatchGenerator.rollIsFileSample().
        boolean decksOnly = Boolean.parseBoolean(
                System.getProperty("sealed.decks.only", "false"));

        int workerIndex = parseIntProp(
                "effect.worker.index", 0, v -> v >= 0, "must be >= 0");

        // The supervisor recycles the longest-running worker every status interval
        // and restarts anything that crashes, so one run is hundreds of JVM
        // lifetimes per worker slot. Every counter the shard writer keeps restarts
        // at zero with the JVM, so the shard token has to name the lifetime as well
        // as the slot; without it a run's record and game ids repeat once per
        // restart and unrelated games merge under one game_id. Minted here rather
        // than required from the launcher so no spawn site can omit it; the
        // property exists only so a test can pin the value.
        String workerLifetime = System.getProperty("effect.worker.lifetime");
        if (workerLifetime == null) {
            workerLifetime = RecordShardWriter.mintLifetime();
        } else if (!RecordShardWriter.isValidLifetime(workerLifetime)) {
            System.err.println(
                    "Error: -Deffect.worker.lifetime must match [0-9a-z]{1,16}, got: "
                            + workerLifetime);
            System.exit(2);
        }

        return new WorkerConfig(
                outputFileProp == null ? null : Path.of(outputFileProp),
                runId, bestOf, sideAProp, sideBProp, sideBWeight,
                effectRecordsProp == null ? null : Path.of(effectRecordsProp),
                workerIndex, workerLifetime, decksOnly);
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
        // Stage four's variant scripts are staged into Forge's custom-cards
        // directory before the card database is read; a script that appears
        // afterwards is invisible for the life of the JVM.
        String variantScripts = System.getProperty("effect.variant.scripts");
        ForgeEnvironmentInitializer.initialize(
                variantScripts == null ? null : Path.of(variantScripts));
        System.out.println("Forge initialized. Starting match generation.");
        System.out.println("Match format: best-of-" + config.bestOf());
    }

    private static void runForever(WorkerConfig config) {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        GeneratedDecksIndex sideAIndex = loadIndex(config.sideADecksFile(), "side-A");
        GeneratedDecksIndex sideBIndex = loadIndex(config.sideBDecksFile(), "side-B");

        // In records-only mode neither sealed writer is constructed at all —
        // not constructed-and-unused, so there is no path by which a coverage
        // or variant run can append to the sealed corpus.
        MatchResultWriter writer = config.recordsOnly()
                ? null : new MatchResultWriter(config.outputFile());
        CardsPlayedWriter cardsPlayedWriter = config.recordsOnly()
                ? null : new CardsPlayedWriter(cardsPlayedPath(config.outputFile()));

        RecordShardWriter effectRecords = null;
        if (config.collectsEffectRecords()) {
            effectRecords = new RecordShardWriter(
                    config.effectRecordsDir(), config.runId(), config.workerIndex(),
                    config.workerLifetime());
            // The tier vector is echoed beside the mode because both are
            // properties of the whole shard that nothing in a record's content
            // reveals: a run collected at a different depth is not comparable
            // with this one, and the depth is what its size is mostly made of.
            System.out.println(
                    "Effect records: " + effectRecords.path()
                            + " [mode=" + AttributionMode.detect().wireValue()
                            + ", tiers=" + PatchedCollectors.CollectionCaps
                                    .fromSystemProperties().snapshotTiers() + "]");
            // The mode is one hook's answer, so a partly-applied patch still
            // reads "patched" while a channel this run meant to collect is
            // quietly empty. This is what names that.
            System.out.println(PatchHooks.report());
            // The worker loops until the supervisor terminates it, so close()
            // is never reached on the normal path. Without this the block in
            // flight — up to a few hundred records — is lost on every stop.
            final RecordShardWriter toFlush = effectRecords;
            Runtime.getRuntime().addShutdownHook(new Thread(toFlush::close));
        }

        // The public 7-arg constructor has no decksOnly parameter (it must
        // keep working unchanged for `sealed match-outcomes`), so a decks-only
        // run goes through the full constructor directly, supplying the same
        // random source and exclusion set the shorter constructors default to.
        MatchGenerator generator = new MatchGenerator(
                eligibleSets, new DeckBuilder(),
                new GamePlayer(config.bestOf(), effectRecords), config.runId(),
                sideAIndex, sideBIndex, config.sideBWeight(),
                MyRandom.getRandom(), MatchGenerator.excludedCardsFromProperties(),
                config.decksOnly());

        long count = 0;
        while (true) {
            try {
                MatchGenerationResult result = generator.generateMatch();
                if (writer != null) {
                    writer.write(result.matchResult());
                }
                if (cardsPlayedWriter != null) {
                    for (CardsPlayedRow row : result.cardsPlayedRows()) {
                        cardsPlayedWriter.write(row);
                    }
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
