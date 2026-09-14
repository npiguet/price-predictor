package com.pricepredictor.connector;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

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

    private static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    private static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    @AfterEach
    void clearSystemProperties() {
        // System properties are JVM-global and Surefire reuses one fork across
        // every test class by default, so a property left set here would leak
        // into whichever test class runs next.
        System.clearProperty("output.file");
        System.clearProperty("effect.records.dir");
        System.clearProperty("match.run.id");
        System.clearProperty("sealed.progress.file");
    }

    // ── parseConfig(): -Dsealed.progress.file ──────────────────────────────

    @Test
    void progressFilePropertyIsRead() {
        System.setProperty("output.file", "dummy-output.txt");
        System.setProperty("match.run.id", RUN_ID);
        System.setProperty("sealed.progress.file", "dummy-progress.txt");

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertEquals(Path.of("dummy-progress.txt"), config.progressFile());
    }

    @Test
    void progressFilePropertyDefaultsToNullWhenAbsent() {
        System.setProperty("output.file", "dummy-output.txt");
        System.setProperty("match.run.id", RUN_ID);

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertNull(config.progressFile());
    }

    @Test
    void progressFileIsReadableInRecordsOnlyMode() {
        // effect.records.dir rather than output.file: records-only mode, the
        // exact configuration collect-coverage / collect-variants runs under.
        // The progress property must not depend on output.file being set --
        // that independence is the fix for Task 4's original defect (a
        // records-only round's progress file never grew).
        System.setProperty("match.run.id", RUN_ID);
        System.setProperty("effect.records.dir", "dummy-records");
        System.setProperty("sealed.progress.file", "dummy-progress.txt");

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertTrue(config.recordsOnly());
        assertEquals(Path.of("dummy-progress.txt"), config.progressFile());
    }

    // ── recordMatch(): the progress line rides no writer's gate ────────────

    private static List<String> deck40(String prefix) {
        List<String> cards = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cards.add(prefix + "Card" + i);
        }
        return cards;
    }

    private static CardsPlayedRow cardsPlayedRow() {
        return new CardsPlayedRow(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                List.of("X"), List.of("Y"), List.of(), List.of(), 'A', 'B');
    }

    /** A two-game match result, built without Forge or a live game. */
    private static MatchGenerationResult twoGameResult() {
        MatchResult matchResult = new MatchResult(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                deck40("A"), deck40("B"), "AA", "BA", 47);
        List<CardsPlayedRow> rows = List.of(cardsPlayedRow(), cardsPlayedRow());
        return new MatchGenerationResult(matchResult, rows);
    }

    @Test
    void progressLineIsWrittenInRecordsOnlyMode(@TempDir Path tmp) throws IOException {
        // Both sealed writers null -- exactly records-only mode -- and the
        // progress line must still be written. This is the exact property
        // whose absence was the round-4 review's CRITICAL finding: a
        // records-only round's progress file never grew, so should_stop(0)
        // was 0 >= matches, false forever, and the round never ended.
        Path progressFile = tmp.resolve("run.progress.txt");
        ProgressWriter progressWriter = new ProgressWriter(progressFile);

        MatchWorkerMain.recordMatch(twoGameResult(), null, null, progressWriter);

        assertEquals(1, Files.readAllLines(progressFile).size());
    }

    @Test
    void progressLineIsWrittenAlongsideBothSealedWriters(@TempDir Path tmp) throws IOException {
        // The non-records-only path (plain `sealed match-outcomes`) must keep
        // writing both sealed outputs exactly as before, with the progress
        // line as a genuine addition rather than a replacement.
        Path outcomesFile = tmp.resolve("match-outcomes.txt");
        Path cardsPlayedFile = tmp.resolve("cards-played.txt");
        Path progressFile = tmp.resolve("run.progress.txt");
        MatchResultWriter writer = new MatchResultWriter(outcomesFile);
        CardsPlayedWriter cardsPlayedWriter = new CardsPlayedWriter(cardsPlayedFile);
        ProgressWriter progressWriter = new ProgressWriter(progressFile);

        MatchWorkerMain.recordMatch(twoGameResult(), writer, cardsPlayedWriter, progressWriter);

        assertEquals(1, Files.readAllLines(outcomesFile).size());
        assertEquals(2, Files.readAllLines(cardsPlayedFile).size());
        assertEquals(1, Files.readAllLines(progressFile).size());
    }

    @Test
    void noProgressWriterMeansNoProgressFileCreated(@TempDir Path tmp) {
        // The unset case (plain `sealed match-outcomes` today, no
        // -Dsealed.progress.file passed): recordMatch(..., null) must be a
        // silent no-op for progress, not an error and not a stray file.
        Path progressFile = tmp.resolve("run.progress.txt");

        assertDoesNotThrow(() -> MatchWorkerMain.recordMatch(twoGameResult(), null, null, null));
        assertFalse(Files.exists(progressFile));
    }

    @Test
    void repeatedCallsAccumulateOneLineEach(@TempDir Path tmp) throws IOException {
        Path progressFile = tmp.resolve("run.progress.txt");
        ProgressWriter progressWriter = new ProgressWriter(progressFile);

        for (int i = 0; i < 5; i++) {
            MatchWorkerMain.recordMatch(twoGameResult(), null, null, progressWriter);
        }

        assertEquals(5, Files.readAllLines(progressFile).size());
    }
}
