package com.pricepredictor.connector.effects;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;

/**
 * One worker's append-only JSONL shard.
 *
 * <p>{@code {run_id}.{worker}.jsonl}, one record per line. A shard per worker
 * rather than one shared file because records are far larger than a
 * match-outcome row, so concurrent appends would interleave mid-record. Line
 * oriented because these JVMs are expected to crash mid-write, and a trailing
 * partial line is exactly what every reader on the Python side skips.
 *
 * <p>Owns the two counters the record ids are built from. Both carry the worker
 * index, because workers count independently and their ids would otherwise
 * collide across a run's shards.
 */
public final class RecordShardWriter implements AutoCloseable {

    private final String runId;
    private final int worker;
    private final Path path;
    private BufferedWriter out;
    private long recordCounter;
    private long gameCounter;

    public RecordShardWriter(Path directory, String runId, int worker) {
        this.runId = runId;
        this.worker = worker;
        this.path = directory.resolve(runId + "." + worker + ".jsonl");
        try {
            Files.createDirectories(directory);
            this.out = Files.newBufferedWriter(
                    path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot open effect-record shard " + path, e);
        }
    }

    public Path path() {
        return path;
    }

    /** Start a new game; returns its {@code game_id}. */
    public String nextGameId() {
        return runId + "." + worker + "." + gameCounter++;
    }

    /** The next {@code record_id}, unique across this run's shards. */
    public String nextRecordId() {
        return runId + "." + worker + "." + recordCounter++;
    }

    public String runId() {
        return runId;
    }

    /** ISO 8601 UTC, the timestamp format every corpus in this repo uses. */
    public static String timestamp() {
        return Instant.now().toString();
    }

    /**
     * Append one already-rendered record line and flush.
     *
     * <p>Flushing per record rather than per game: a crash loses at most the
     * line being written, which the readers already tolerate, instead of losing
     * a game's worth of records that were never the crash's fault.
     */
    public synchronized void write(String recordJson) {
        try {
            out.write(recordJson);
            out.newLine();
            out.flush();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot append to " + path, e);
        }
    }

    @Override
    public synchronized void close() {
        if (out == null) {
            return;
        }
        try {
            out.close();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot close " + path, e);
        } finally {
            out = null;
        }
    }
}
