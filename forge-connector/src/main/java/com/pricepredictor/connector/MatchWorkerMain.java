package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.AttributionMode;
import com.pricepredictor.connector.effects.PatchHooks;
import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.RecordShardWriter;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

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
 *   <li>{@code -Deffect.random.seat.share=<F>} and
 *       {@code -Deffect.random.seat.probability=<P>} — in a share {@code F} of
 *       matches, one seat chosen per match is a {@link RandomSeatController}
 *       that draws uniformly from the legal options with probability {@code P}
 *       at each decision point (FR-021). Such a match writes effect records
 *       and its progress line, never a sealed row (FR-025). Read with the
 *       other {@code effect.*} caps; echoed at startup when the share is
 *       above zero.</li>
 *   <li>{@code -Dsealed.progress.file=<path>} — when set, one line is appended
 *       to this file per completed match via {@link ProgressWriter}, so a
 *       caller that never touches {@code output.file} (records-only mode) can
 *       still bound a run by counting matches played. Written unconditionally
 *       — independent of {@link WorkerConfig#recordsOnly()} and of both sealed
 *       writers — because a progress tally is not a corpus and must not ride
 *       the gate that keeps records-only mode out of one.</li>
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
            boolean decksOnly,
            Path progressFile) {

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

        String runId = WorkerProperties.require("match.run.id");

        int bestOf = WorkerProperties.bestOf("match.best.of", DEFAULT_BEST_OF);

        String sideAProp = System.getProperty("side.a.decks.file");
        String sideBProp = System.getProperty("side.b.decks.file");
        String sideBWeightProp = System.getProperty("side.b.decks.weight");

        if (sideBProp == null && sideBWeightProp != null) {
            System.err.println(
                    "Error: -Dside.b.decks.weight is only valid with -Dside.b.decks.file");
            System.exit(2);
        }

        int sideBWeight = WorkerProperties.intProperty(
                "side.b.decks.weight", DEFAULT_SIDE_B_WEIGHT,
                v -> v >= 1, "must be >= 1");

        // A coverage or variant deck belongs to no set, so the roll's
        // Forge-method branch would resolve its sentinel set code against
        // Forge's booster/edition tables and throw. This forces the roll to
        // the decks file every time. See MatchGenerator.rollIsFileSample().
        boolean decksOnly = WorkerProperties.flag("sealed.decks.only");

        int workerIndex = WorkerProperties.intProperty(
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

        // A counter, not a corpus: read independent of every other property
        // here, and never validated against output.file / effect.records.dir,
        // because a progress tally must be obtainable even in records-only
        // mode -- that independence is the whole point (see the class-level
        // Javadoc entry for this property).
        String progressFileProp = System.getProperty("sealed.progress.file");

        return new WorkerConfig(
                outputFileProp == null ? null : Path.of(outputFileProp),
                runId, bestOf, sideAProp, sideBProp, sideBWeight,
                effectRecordsProp == null ? null : Path.of(effectRecordsProp),
                workerIndex, workerLifetime, decksOnly,
                progressFileProp == null ? null : Path.of(progressFileProp));
    }

    private static void initializeForge(WorkerConfig config) {
        System.out.println("Initializing Forge environment...");
        // Stage four's variant scripts are already in Forge's custom-cards
        // directory: VariantSidecarMain put them there, in one process, before
        // the first worker started. This one registers their names and does
        // not copy. A dozen workers each copying thousands of scripts into
        // that one shared directory while the others walk it during
        // FModel.initialize is a race, and a worker that reads it half-written
        // has a card database missing the very variants its decks name.
        String variantScripts = System.getProperty("effect.variant.scripts");
        ForgeEnvironmentInitializer.initialize(
                variantScripts == null ? null : Path.of(variantScripts), false);
        System.out.println("Forge initialized. Starting match generation.");
        System.out.println("Match format: best-of-" + config.bestOf());
    }

    private static void runForever(WorkerConfig config) {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        System.out.println("Eligible sets: " + eligibleSets.size());

        MatchGenerator.DeckSource sideA = loadDecks(config.sideADecksFile(), "side-A");
        MatchGenerator.DeckSource sideB = loadDecks(config.sideBDecksFile(), "side-B");

        // In records-only mode neither sealed writer is constructed at all —
        // not constructed-and-unused, so there is no path by which a coverage
        // or variant run can append to the sealed corpus.
        MatchResultWriter writer = config.recordsOnly()
                ? null : new MatchResultWriter(config.outputFile());
        CardsPlayedWriter cardsPlayedWriter = config.recordsOnly()
                ? null : new CardsPlayedWriter(cardsPlayedPath(config.outputFile()));
        // Never gated by recordsOnly(): a progress tally is not a corpus, and
        // gating it on the same guard that keeps records-only mode out of the
        // sealed corpus is exactly the bug this property exists to fix (a
        // records-only round's progress file would otherwise never grow).
        ProgressWriter progressWriter = config.progressFile() == null
                ? null : new ProgressWriter(config.progressFile());

        RecordShardWriter effectRecords = null;
        if (config.collectsEffectRecords()) {
            effectRecords = new RecordShardWriter(
                    config.effectRecordsDir(), config.runId(), config.workerIndex(),
                    config.workerLifetime());
            System.out.println(
                    "Effect records: " + effectRecords.path()
                            + " [mode=" + AttributionMode.detect().wireValue() + "]");
            // The mode is one hook's answer, so a partly-applied patch still
            // reads "patched" while a channel this run meant to collect is
            // quietly empty. This is what names that.
            System.out.println(PatchHooks.report());
            PatchedCollectors.CollectionCaps caps =
                    PatchedCollectors.CollectionCaps.fromSystemProperties();
            if (caps.randomSeatEnabled()) {
                // Named at startup the way the probe state is: a run whose
                // random-seat matches wrote no sealed rows is a run someone
                // asked for, and the log is where that is checked.
                System.out.println("Random seat: share=" + caps.randomSeatShare()
                        + ", probability=" + caps.randomSeatProbability()
                        + " (random-seat matches write effect records only)");
            }
            // The worker loops until the supervisor terminates it, so close()
            // is never reached on the normal path. Without this the block in
            // flight — up to a few hundred records — is lost on every stop.
            final RecordShardWriter toFlush = effectRecords;
            Runtime.getRuntime().addShutdownHook(new Thread(toFlush::close));
        }

        SilencedHookReport silencedHookReport = new SilencedHookReport();
        MatchGenerator generator = MatchGenerator.builder(eligibleSets, config.runId())
                .gamePlayer(new GamePlayer(config.bestOf(), effectRecords))
                .sideA(sideA)
                .sideB(sideB, config.sideBWeight())
                .excludedCards(HoldoutCards.fromSystemProperty())
                .decksOnly(config.decksOnly())
                .build();

        WorkerLoop.runUntilBroken(
                () -> {
                    recordMatch(generator.generateMatch(),
                            writer, cardsPlayedWriter, progressWriter);
                    silencedHookReport.afterMatch();
                    return WorkerLoop.Outcome.COMPLETED;
                },
                new WorkerLoop.Progress(
                        System.out, 10, "matches generated", "Error generating match"));
    }

    /**
     * Record one completed match to whichever of the three writers are wired
     * up. Factored out of {@link #runForever}'s infinite loop so the
     * independence this method's contract depends on — the progress line is
     * written whether or not {@code writer}/{@code cardsPlayedWriter} are
     * {@code null} — is exercisable by a plain unit test that constructs a
     * {@link MatchGenerationResult} fixture and never touches Forge.
     */
    /**
     * Names, once per match boundary, each effect-record hook that has gone
     * quiet since the last one.
     *
     * <p>The engine contains a listener that throws, reports it once on
     * stderr and stops listening to it, so a channel can empty mid-run while
     * the worker goes on playing. That one stderr line scrolls away; this
     * line repeats the channel's name where the match log is read.
     */
    static final class SilencedHookReport {
        private int reported;

        void afterMatch() {
            List<String> silenced = PatchHooks.silencedHooks();
            if (silenced.size() > reported) {
                System.err.println("effect-record hooks silenced after a listener failure, "
                        + "their channels now empty: " + silenced);
                reported = silenced.size();
            }
        }
    }

    static void recordMatch(
            MatchGenerationResult result,
            MatchResultWriter writer,
            CardsPlayedWriter cardsPlayedWriter,
            ProgressWriter progressWriter) {
        // Records-only per match (FR-025), beside the per-worker mode the
        // writers' nulls express: a random-seat match had one seat playing
        // at random, so its outcome says nothing about either deck and
        // neither sealed corpus may take a row from it. Its effect records
        // were written as the games ran; only the progress line is left.
        boolean sealedRows = !result.randomSeat();
        if (writer != null && sealedRows) {
            writer.write(result.matchResult());
        }
        if (cardsPlayedWriter != null && sealedRows) {
            for (CardsPlayedRow row : result.cardsPlayedRows()) {
                cardsPlayedWriter.write(row);
            }
        }
        // Unconditional on writer/cardsPlayedWriter: a records-only round
        // (both null) must still advance its progress file, or a bounded
        // round built on this worker can never observe that it is done.
        if (progressWriter != null) {
            progressWriter.write();
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

    /** Forge-built decks when no file is named; exits 2 on a file that does not load. */
    private static MatchGenerator.DeckSource loadDecks(String pathProp, String label) {
        if (pathProp == null) {
            return MatchGenerator.DeckSource.FORGE_BUILT;
        }
        Path path = Path.of(pathProp);
        System.out.println("Loading " + label + " decks from " + path);
        try {
            GeneratedDecksIndex index = GeneratedDecksIndex.load(path);
            System.out.println("Loaded " + index.size() + " " + label + " decks.");
            return new MatchGenerator.DeckSource.FileSampled(index);
        } catch (Exception e) {
            System.err.println(
                    "Error loading " + label + " decks file: " + e.getMessage());
            System.exit(2);
            return null;  // unreachable
        }
    }
}
