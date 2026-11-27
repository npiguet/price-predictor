package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.cardsPlayedRow;
import static com.pricepredictor.connector.MatchFixtures.matchResult;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MatchWorkerMain}'s pure config-parsing and per-match
 * recording helpers.
 *
 * <p>No Forge initialization needed for either half: {@code parseConfig()}
 * only reads JVM system properties (no {@link ForgeEnvironmentInitializer}
 * call), and {@code recordMatch()} only routes an already-built
 * {@link MatchGenerationResult} fixture to writers -- neither touches a live
 * game, so this class carries no {@code @Tag("integration")} and runs under
 * the default {@code mvn test}. {@code runForever}'s own infinite loop is the
 * one seam this file does not reach: it needs a live {@link MatchGenerator}
 * driving real Forge games, which is exactly what {@code recordMatch} was
 * factored out of the loop to avoid requiring.
 */
class MatchWorkerMainTest {

    /** The system properties as they stood before the test. */
    private Properties before;

    @BeforeEach
    void snapshotSystemProperties() {
        before = (Properties) System.getProperties().clone();
    }

    /**
     * System properties are JVM-global and Surefire reuses one fork across
     * every test class by default, so whatever a test set is put back: a
     * property left set here would leak into whichever test class runs next.
     */
    @AfterEach
    void restoreSystemProperties() {
        System.getProperties().keySet().removeIf(key -> !before.containsKey(key));
        before.forEach((key, value) -> System.getProperties().put(key, value));
    }

    // ── parseConfig(): the graceful stop ────────────────────────────────

    private static void recordsOnly() {
        System.setProperty("effect.records.dir", "dummy-records");
        System.setProperty("match.run.id", RUN_ID);
    }

    /** A recycle time of zero would stop before every first match: refused at parse. */
    @Test
    void aRecycleTimeOfZeroIsRefusedBeforeForgeStarts() {
        recordsOnly();
        System.setProperty("worker.recycle.seconds", "0");

        ToolMain.Exit exit = assertThrows(ToolMain.Exit.class, MatchWorkerMain::parseConfig);

        assertEquals(ExitCode.BAD_CONFIG, exit.code());
    }

    @Test
    void aMalformedRecycleTimeIsRefusedBeforeForgeStarts() {
        recordsOnly();
        System.setProperty("worker.recycle.seconds", "soon");

        assertEquals(ExitCode.BAD_CONFIG,
                assertThrows(ToolMain.Exit.class, MatchWorkerMain::parseConfig).code());
    }

    /**
     * A listed rule's own directed-fork probability that is not a number in
     * [0, 1] is refused before Forge starts, the property named, rather than
     * defaulted: a run sets it to steer what the rule collects.
     */
    @ParameterizedTest
    @ValueSource(strings = {"1.5", "-0.1", "NaN", "often"})
    void aMalformedRuleProbabilityIsRefusedBeforeForgeStarts(String value) {
        recordsOnly();
        System.setProperty("effect.directed.fork.rules", "118.12b,704.5f");
        System.setProperty("effect.directed.probability.704.5f", value);
        PrintStream err = System.err;
        ByteArrayOutputStream printed = new ByteArrayOutputStream();
        System.setErr(new PrintStream(printed, true, StandardCharsets.UTF_8));
        ToolMain.Exit exit;
        try {
            exit = assertThrows(ToolMain.Exit.class, MatchWorkerMain::parseConfig);
        } finally {
            System.setErr(err);
        }

        assertEquals(ExitCode.BAD_CONFIG, exit.code());
        assertTrue(printed.toString(StandardCharsets.UTF_8)
                .contains("-Deffect.directed.probability.704.5f "), printed.toString(StandardCharsets.UTF_8));
    }

    /** A rule's own probability is read for the rules listed; the others take the run's. */
    @Test
    void aListedRuleTakesItsOwnProbabilityAndTheOthersTheRuns() {
        recordsOnly();
        System.setProperty("effect.directed.fork.rules", "118.12b,704.5f");
        System.setProperty("effect.directed.fork.probability", "0.25");
        System.setProperty("effect.directed.probability.704.5f", "1");
        System.setProperty("effect.directed.probability.510.2", "0");

        CollectionCaps caps = MatchWorkerMain.parseConfig().caps();

        assertEquals(1.0, caps.directedForkProbability("704.5f"));
        assertEquals(0.25, caps.directedForkProbability("118.12b"));
        assertEquals(Map.of("704.5f", 1.0), caps.directedRuleProbabilities(),
                "a rule the run does not list is not read");
    }

    /** The recycle time counts from the worker's start, not from the end of Forge's. */
    @Test
    void theRecycleTimeCountsFromTheWorkersStart() {
        recordsOnly();
        System.setProperty("worker.recycle.seconds", "5");

        MatchWorkerMain.WorkerConfig config =
                MatchWorkerMain.parseConfig(System.nanoTime() - 6_000_000_000L);

        assertEquals(java.util.Optional.of("recycling after 5 s"), config.stop().reason());
    }

    /** A stop file there at startup asks the worker to stop before its first match. */
    @Test
    void aStopFileThereAtStartupIsAStopRequest(@org.junit.jupiter.api.io.TempDir Path dir)
            throws java.io.IOException {
        recordsOnly();
        Path stop = java.nio.file.Files.createFile(dir.resolve("worker.stop"));
        System.setProperty("worker.stop.file", stop.toString());

        assertEquals(java.util.Optional.of("stop file " + stop),
                MatchWorkerMain.parseConfig().stop().reason());
    }

    // ── parseConfig(): -Dsealed.progress.file ──────────────────────────────

    @Test
    void progressFilePropertyIsEmptyWhenAbsent() {
        System.setProperty("output.file", "dummy-output.txt");
        System.setProperty("match.run.id", RUN_ID);

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertTrue(config.progressFile().isEmpty());
    }

    /**
     * A blank optional path is no path: read as given, a blank
     * {@code -Doutput.file=} would name the empty path and take the worker
     * out of records-only mode.
     */
    @Test
    void blankOptionalPathsReadAsAbsent() {
        System.setProperty("output.file", " ");
        System.setProperty("effect.records.dir", "dummy-records");
        System.setProperty("match.run.id", RUN_ID);
        System.setProperty("sealed.progress.file", "");

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertTrue(config.recordsOnly());
        assertTrue(config.progressFile().isEmpty());
        assertEquals(Optional.of(Path.of("dummy-records")), config.effectRecordsDir());
    }

    @Test
    void progressFileIsReadableInRecordsOnlyMode() {
        // effect.records.dir rather than output.file: records-only mode, the
        // exact configuration collect-coverage / collect-variants runs under.
        // The progress property must not depend on output.file being set --
        // without that independence a records-only round's progress file
        // never grows.
        System.setProperty("match.run.id", RUN_ID);
        System.setProperty("effect.records.dir", "dummy-records");
        System.setProperty("sealed.progress.file", "dummy-progress.txt");

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertTrue(config.recordsOnly());
        assertEquals(Optional.of(Path.of("dummy-progress.txt")), config.progressFile());
    }

    // ── recordMatch(): the progress line rides no corpus's gate ────────────

    /** A two-game match result, built without Forge or a live game. */
    private static MatchGenerationResult twoGameResult() {
        return new MatchGenerationResult(
                matchResult(), List.of(cardsPlayedRow(), cardsPlayedRow()), false);
    }

    @Test
    void progressLineIsWrittenInRecordsOnlyMode(@TempDir Path tmp) throws IOException {
        // No sealed corpus -- exactly records-only mode -- and the progress
        // line must still be written: without it a records-only round's
        // progress file never grows, should_stop(0) reads 0 >= matches,
        // false forever, and the round never ends.
        Path progressFile = tmp.resolve("run.progress.txt");

        MatchWorkerMain.recordMatch(twoGameResult(), Optional.empty(),
                Optional.of(new ProgressWriter(progressFile)));

        assertEquals(1, Files.readAllLines(progressFile).size());
    }

    @Test
    void progressLineIsWrittenAlongsideTheSealedCorpus(@TempDir Path tmp) throws IOException {
        // The non-records-only path (plain `sealed match-outcomes`) must keep
        // writing both sealed outputs exactly as before, with the progress
        // line as a genuine addition rather than a replacement.
        Path outcomesFile = tmp.resolve("match-outcomes.txt");
        Path cardsPlayedFile = tmp.resolve("cards-played.txt");
        Path progressFile = tmp.resolve("run.progress.txt");

        MatchWorkerMain.recordMatch(twoGameResult(),
                Optional.of(SealedCorpus.beside(outcomesFile)),
                Optional.of(new ProgressWriter(progressFile)));

        assertEquals(1, Files.readAllLines(outcomesFile).size());
        assertEquals(2, Files.readAllLines(cardsPlayedFile).size());
        assertEquals(1, Files.readAllLines(progressFile).size());
    }

    /**
     * 024 FR-025: a match with a random seat writes its progress line and nothing
     * to either sealed file, even with the sealed corpus wired (Feature 024 story 3
     * scenario 5). Its effect records were written as the games ran; the
     * outcome of a game one seat played at random says nothing about a deck.
     */
    @Test
    void aRandomSeatMatchAddsNoOutcomeOrCardsPlayedRow(@TempDir Path tmp) throws IOException {
        Path outcomesFile = tmp.resolve("match-outcomes.txt");
        Path cardsPlayedFile = tmp.resolve("cards-played.txt");
        Path progressFile = tmp.resolve("run.progress.txt");
        MatchGenerationResult ordinary = twoGameResult();
        MatchGenerationResult randomSeat = new MatchGenerationResult(
                ordinary.matchResult(), ordinary.cardsPlayedRows(), true);

        MatchWorkerMain.recordMatch(randomSeat,
                Optional.of(SealedCorpus.beside(outcomesFile)),
                Optional.of(new ProgressWriter(progressFile)));

        assertFalse(Files.exists(outcomesFile), "no match-outcomes row for a random-seat match");
        assertFalse(Files.exists(cardsPlayedFile), "no cards-played row either");
        assertEquals(1, Files.readAllLines(progressFile).size(), "the progress line is written");
    }

    /** The result still validates its rows for an ordinary match, and tolerates none for a random-seat one. */
    @Test
    void aRandomSeatResultMayCarryNoRows() {
        MatchResult matchResult = twoGameResult().matchResult();
        assertDoesNotThrow(() -> new MatchGenerationResult(matchResult, List.of(), true));
        assertThrows(IllegalArgumentException.class,
                () -> new MatchGenerationResult(matchResult, List.of(), false));
    }

    @Test
    void noProgressFileIsASilentNoOp() {
        // No -Dsealed.progress.file passed: recordMatch skips the progress
        // line rather than failing.
        assertDoesNotThrow(() -> MatchWorkerMain.recordMatch(
                twoGameResult(), Optional.empty(), Optional.empty()));
    }

}
