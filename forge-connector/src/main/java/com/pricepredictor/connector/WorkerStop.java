package com.pricepredictor.connector;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.function.LongSupplier;

/**
 * Whether a worker is to stop at the next game end, and why: the graceful
 * stop the supervisor asks for, or the worker's own recycling.
 *
 * <p>A worker killed mid-game cuts that game: its records already written stay
 * in the shard, and the end-of-game ones — the reservoirs' trigger and mana
 * records, the game-end marker — never come. Asked between games and before
 * each match, a stop lets the game in progress end normally, its records and
 * marker written, and the shard closed whole. The supervisor asks without a
 * console signal, which Windows cannot deliver to one JVM of a process group:
 * it creates the file named by {@value #STOP_FILE}. A worker never creates or
 * deletes that file; its existence is the request, its content is ignored.
 *
 * <p>{@value #RECYCLE_SECONDS} is the worker's own recycling: once its JVM has
 * run that long, it stops at the next game end the same way, so the pool
 * restarts a fresh JVM without killing one mid-game.
 */
final class WorkerStop {

    /** {@code -D} property naming the stop file; absent, no file is watched. */
    static final String STOP_FILE = "worker.stop.file";
    /** {@code -D} property: stop at the first game end after this many seconds; absent, never. */
    static final String RECYCLE_SECONDS = "worker.recycle.seconds";

    /** A worker that is never asked to stop. */
    static final WorkerStop NEVER = new WorkerStop(null, -1, System::nanoTime);

    private final Path stopFile;
    private final long recycleNanos;
    private final LongSupplier clock;
    private final long startedNanos;

    WorkerStop(Path stopFile, long recycleSeconds, LongSupplier clock) {
        this.stopFile = stopFile;
        this.recycleNanos = recycleSeconds < 0 ? -1 : recycleSeconds * 1_000_000_000L;
        this.clock = clock;
        this.startedNanos = clock.getAsLong();
    }

    /**
     * The stop the JVM's system properties ask for. A malformed
     * {@value #RECYCLE_SECONDS} ends the worker with
     * {@link ExitCode#BAD_CONFIG}.
     */
    static WorkerStop fromSystemProperties() {
        Path file = WorkerProperties.optionalPath(STOP_FILE).orElse(null);
        long seconds = WorkerProperties.optional(RECYCLE_SECONDS).isEmpty() ? -1
                : WorkerProperties.intProperty(RECYCLE_SECONDS, -1,
                        WorkerProperties.OnMalformed.EXIT, v -> v >= 0, "must be >= 0");
        return new WorkerStop(file, seconds, System::nanoTime);
    }

    /** Why the worker is to stop now, or empty to go on. */
    Optional<String> reason() {
        if (stopFile != null && Files.exists(stopFile)) {
            return Optional.of("stop file " + stopFile);
        }
        if (recycleNanos >= 0 && clock.getAsLong() - startedNanos >= recycleNanos) {
            return Optional.of("recycling after " + recycleNanos / 1_000_000_000L + " s");
        }
        return Optional.empty();
    }

    /** What the worker prints at startup about its stop, or empty when it has none. */
    Optional<String> describe() {
        if (stopFile == null && recycleNanos < 0) {
            return Optional.empty();
        }
        return Optional.of("Graceful stop: at a game end, when " + (stopFile == null ? ""
                : stopFile + " exists" + (recycleNanos < 0 ? "" : " or ")) + (recycleNanos < 0 ? ""
                : "the JVM has run " + recycleNanos / 1_000_000_000L + " s"));
    }

    /**
     * Thrown at a game end when the worker is to stop: the match in progress
     * is left there, unrecorded as a match, and the worker loop ends.
     */
    static final class Requested extends RuntimeException {
        private final String reason;

        Requested(String reason) {
            super("graceful stop: " + reason, null, false, false);
            this.reason = reason;
        }

        String reason() {
            return reason;
        }
    }
}
