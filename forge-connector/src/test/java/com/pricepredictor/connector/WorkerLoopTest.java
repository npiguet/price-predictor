package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

import static com.pricepredictor.connector.WorkerLoop.Outcome.ABANDONED;
import static com.pricepredictor.connector.WorkerLoop.Outcome.COMPLETED;
import static com.pricepredictor.connector.WorkerLoop.Outcome.SKIPPED;
import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link WorkerLoop}'s consecutive-failure cap: a worker whose every attempt
 * fails must give up rather than spin, and a worker that mostly works must
 * never reach the cap.
 */
class WorkerLoopTest {

    private final ByteArrayOutputStream progressBytes = new ByteArrayOutputStream();
    private final WorkerLoop.Progress progress = new WorkerLoop.Progress(
            new PrintStream(progressBytes, true, StandardCharsets.UTF_8),
            2, "things done", "Error doing thing");

    /**
     * Plays a script of outcomes, a {@code null} entry throwing, then throws
     * forever. Counts its calls so a test can assert the loop did not stop early.
     */
    private static final class Script implements WorkerLoop.Attempt {
        private final WorkerLoop.Outcome[] script;
        int calls = 0;

        Script(WorkerLoop.Outcome... script) {
            this.script = script;
        }

        @Override
        public WorkerLoop.Outcome run() {
            int index = calls++;
            WorkerLoop.Outcome outcome = index < script.length ? script[index] : null;
            if (outcome == null) {
                throw new IllegalStateException("scripted failure " + calls);
            }
            return outcome;
        }
    }

    private long run(Script script, int cap) {
        return assertTimeoutPreemptively(Duration.ofSeconds(10),
                () -> WorkerLoop.run(script, cap, progress));
    }

    @Test
    void aWorkerWhoseEveryAttemptThrowsGivesUpInsteadOfSpinning() {
        // An error thrown on every match once left the progress file flat:
        // `should_stop` never fired and ForgeWorkerPool.run() blocked forever,
        // which is the hang the sentinel-set-code NPE actually produced.
        Script script = new Script();

        assertEquals(0, run(script, 10));
        assertEquals(10, script.calls, "gave up at a different count than asked");
    }

    @Test
    void aSkipCountsAsAFailure() {
        // A draft worker handed a set with no booster template skips every
        // draft without throwing; the cap has to catch that too.
        Script script = new Script(SKIPPED, SKIPPED, SKIPPED, SKIPPED, SKIPPED);

        assertEquals(0, run(script, 3));
        assertEquals(3, script.calls);
    }

    @Test
    void aSuccessResetsTheConsecutiveFailureCount() {
        // Two failures, a success, two failures, a success, then failures
        // forever -- against a cap of 3.
        Script script = new Script(null, SKIPPED, COMPLETED, null, null, COMPLETED);

        assertEquals(2, run(script, 3), "the successes in the middle were not counted");
        assertEquals(9, script.calls,
                "six scripted calls plus the three consecutive failures that "
                        + "finally reached the cap");
    }

    @Test
    void anAbandonedAttemptNeitherCountsNorResets() {
        // The supervisor aborted it: not the worker's failure, not a success.
        Script script = new Script(null, ABANDONED, ABANDONED, null);

        assertEquals(0, run(script, 3));
        assertEquals(5, script.calls,
                "two failures around two abandoned drafts, then the third failure");
    }

    @Test
    void progressIsReportedEveryNCompletedAttempts() {
        Script script = new Script(COMPLETED, COMPLETED, COMPLETED, COMPLETED);

        assertEquals(4, run(script, 1));

        assertEquals("Worker: 2 things done" + System.lineSeparator()
                        + "Worker: 4 things done" + System.lineSeparator(),
                progressBytes.toString(StandardCharsets.UTF_8));
    }
}
