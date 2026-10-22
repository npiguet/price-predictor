package com.pricepredictor.connector;

import java.io.PrintStream;

/**
 * The loop every autonomous worker runs — {@link MatchWorkerMain},
 * {@link DraftWorkerMain}, {@link DraftGameWorkerMain}: attempt one unit of
 * work, repeat until the supervisor kills the process, or until too many
 * attempts in a row produce nothing.
 *
 * <p>The cap is what keeps a broken configuration from spinning forever. The
 * supervisors bound a run by watching output grow, so a worker whose every
 * attempt throws — or quietly returns nothing — leaves that output flat, and
 * the run never ends with nothing to show for it. That is not hypothetical:
 * a sentinel set code once made every match attempt throw, and a draft
 * worker given a set with no booster template skipped every draft in silence.
 */
final class WorkerLoop {

    /**
     * How many attempts may fail in a row, with no success between them,
     * before the worker gives up.
     *
     * <p>Ten. A worker whose every attempt fails does so in milliseconds, so
     * ten attempts cost well under a second and the supervisor restarts the
     * slot at once; a worker whose attempts mostly succeed never reaches ten,
     * because any success resets the count — a run of ten failures with not
     * one success between them is not bad luck, it is a broken configuration.
     */
    static final int MAX_CONSECUTIVE_FAILURES = 10;

    /** What one attempt came to. */
    enum Outcome {
        /** It produced its unit of work; the failure count resets. */
        COMPLETED,
        /**
         * It produced nothing, for a reason it has already logged and that the
         * next attempt will likely meet again: counts as a failure.
         */
        SKIPPED,
        /**
         * It produced nothing because the supervisor said so (an aborted draft):
         * not the worker's failure and not a success, so the count stands.
         */
        ABANDONED
    }

    /** One attempt. An exception counts as a failure, and its stack trace is logged. */
    @FunctionalInterface
    interface Attempt {
        Outcome run() throws Exception;
    }

    /**
     * How a worker names its progress.
     *
     * @param stream     where the progress line goes; stderr for a worker whose
     *                   stdout is a protocol channel
     * @param every      a line every this many completed attempts
     * @param completed  the progress noun, e.g. {@code "matches generated"}
     * @param failure    the error prefix, e.g. {@code "Error generating match"}
     */
    record Progress(PrintStream stream, int every, String completed, String failure) {}

    private WorkerLoop() {
    }

    /**
     * Run attempts until {@code maxConsecutiveFailures} fail in a row.
     *
     * <p>Errors (OOM and friends) are not caught: they propagate untouched and
     * the supervisor restarts the worker.
     *
     * @return how many attempts completed before the cap was reached. On a
     *         healthy worker this never returns — the supervisor kills it.
     */
    static long run(Attempt attempt, int maxConsecutiveFailures, Progress progress) {
        long completed = 0;
        int consecutiveFailures = 0;
        while (true) {
            Outcome outcome;
            try {
                outcome = attempt.run();
            } catch (Exception e) {
                System.err.println(progress.failure() + ": " + e.getMessage());
                e.printStackTrace(System.err);
                outcome = Outcome.SKIPPED;
            }
            switch (outcome) {
                case COMPLETED -> {
                    completed++;
                    consecutiveFailures = 0;
                    if (completed % progress.every() == 0) {
                        progress.stream().println(
                                "Worker: " + completed + " " + progress.completed());
                        progress.stream().flush();
                    }
                }
                case SKIPPED -> {
                    consecutiveFailures++;
                    if (consecutiveFailures >= maxConsecutiveFailures) {
                        return completed;
                    }
                }
                case ABANDONED -> {
                    // Neither: the supervisor decided, and it is counting.
                }
            }
        }
    }

    /**
     * Run attempts until the cap, then say why and exit 1, so the supervisor
     * restarts the slot and the operator sees a failing worker rather than a
     * run that never finishes.
     */
    static void runUntilBroken(Attempt attempt, Progress progress) {
        long completed = run(attempt, MAX_CONSECUTIVE_FAILURES, progress);
        System.err.println("Worker giving up after " + MAX_CONSECUTIVE_FAILURES
                + " consecutive failed attempts with no success between them ("
                + completed + " " + progress.completed() + " in this JVM). Exiting "
                + "non-zero so the supervisor restarts the slot and the operator sees "
                + "a failing worker rather than a run that never finishes.");
        System.err.flush();
        System.exit(1);
    }
}
