package com.pricepredictor.connector;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.TS;
import static com.pricepredictor.connector.MatchFixtures.deck40;
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
        // without that independence a records-only round's progress file
        // never grows.
        System.setProperty("match.run.id", RUN_ID);
        System.setProperty("effect.records.dir", "dummy-records");
        System.setProperty("sealed.progress.file", "dummy-progress.txt");

        MatchWorkerMain.WorkerConfig config = MatchWorkerMain.parseConfig();

        assertTrue(config.recordsOnly());
        assertEquals(Path.of("dummy-progress.txt"), config.progressFile());
    }

    // ── recordMatch(): the progress line rides no writer's gate ────────────

    private static CardsPlayedRow cardsPlayedRow() {
        return new CardsPlayedRow(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                List.of("X"), List.of("Y"), List.of(), List.of(), Side.A, Side.B);
    }

    /** A two-game match result, built without Forge or a live game. */
    private static MatchGenerationResult twoGameResult() {
        MatchResult matchResult = new MatchResult(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                deck40("A"), deck40("B"), "AA", "BA", 47);
        List<CardsPlayedRow> rows = List.of(cardsPlayedRow(), cardsPlayedRow());
        return new MatchGenerationResult(matchResult, rows, false);
    }

    @Test
    void progressLineIsWrittenInRecordsOnlyMode(@TempDir Path tmp) throws IOException {
        // Both sealed writers null -- exactly records-only mode -- and the
        // progress line must still be written: without it a records-only
        // round's progress file never grows, should_stop(0) reads
        // 0 >= matches, false forever, and the round never ends.
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

    /**
     * FR-025: a match with a random seat writes its progress line and nothing
     * to either sealed corpus, even with both writers wired (spec Story 3
     * scenario 5). Its effect records were written as the games ran; the
     * outcome of a game one seat played at random says nothing about a deck.
     */
    @Test
    void aRandomSeatMatchAddsNoOutcomeOrCardsPlayedRow(@TempDir Path tmp) throws IOException {
        Path outcomesFile = tmp.resolve("match-outcomes.txt");
        Path cardsPlayedFile = tmp.resolve("cards-played.txt");
        Path progressFile = tmp.resolve("run.progress.txt");
        MatchResultWriter writer = new MatchResultWriter(outcomesFile);
        CardsPlayedWriter cardsPlayedWriter = new CardsPlayedWriter(cardsPlayedFile);
        ProgressWriter progressWriter = new ProgressWriter(progressFile);
        MatchGenerationResult ordinary = twoGameResult();
        MatchGenerationResult randomSeat = new MatchGenerationResult(
                ordinary.matchResult(), ordinary.cardsPlayedRows(), true);

        MatchWorkerMain.recordMatch(randomSeat, writer, cardsPlayedWriter, progressWriter);

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
    void noProgressWriterIsASilentNoOp() {
        // No -Dsealed.progress.file passed: recordMatch(..., null) skips the
        // progress line rather than failing.
        assertDoesNotThrow(() -> MatchWorkerMain.recordMatch(twoGameResult(), null, null, null));
    }
}
