package com.pricepredictor.connector.effects;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.zip.GZIPOutputStream;

/**
 * One worker's append-only, gzip-compressed JSONL shard.
 *
 * <p>{@code {run_id}.{worker}.jsonl.gz}, one record per line. A shard per worker
 * rather than one shared file because records are far larger than a
 * match-outcome row, so concurrent appends would interleave mid-record.
 *
 * <p>Compressed because the corpus is the run's real cost on disk: a game
 * produces a few hundred records and several megabytes of JSON, so an overnight
 * run fills a disk long before it satisfies gate 2. These records compress
 * roughly tenfold — they are highly repetitive JSON, the same keys and the same
 * board described over and over.
 *
 * <p>The file is a <b>concatenation of complete gzip members</b>, one per block
 * of {@link #BLOCK_RECORDS} records, which is what keeps it appendable and
 * crash-tolerant at the same time. Every member stands alone, so a JVM that dies
 * mid-write truncates the final member and leaves every earlier one readable;
 * readers stop at the truncation exactly as they already skip a trailing partial
 * line. A single gzip stream would have neither property, and a member per
 * record would compress almost nothing.
 *
 * <p>The cost is that a crash loses the block in flight rather than one line.
 * That is bounded by {@code BLOCK_RECORDS} and is worth it: these JVMs crash on
 * long games, but a few hundred records out of hundreds of thousands is noise.
 *
 * <p>Owns the two counters the record ids are built from. Both carry the worker
 * index, because workers count independently and their ids would otherwise
 * collide across a run's shards.
 */
public final class RecordShardWriter implements AutoCloseable {

    /**
     * Records per gzip member.
     *
     * <p>The trade is compression ratio against how much a crash costs. A few
     * hundred records is already past the point where the ratio stops improving,
     * and is a few seconds of one worker's output.
     */
    static final int BLOCK_RECORDS = 256;

    public static final String SUFFIX = ".jsonl.gz";

    private final String runId;
    private final int worker;
    private final Path path;
    private final StringBuilder block = new StringBuilder();
    private int blockRecords;
    private boolean closed;
    private long recordCounter;
    private long gameCounter;

    public RecordShardWriter(Path directory, String runId, int worker) {
        this.runId = runId;
        this.worker = worker;
        this.path = directory.resolve(runId + "." + worker + SUFFIX);
        try {
            Files.createDirectories(directory);
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

    /** Append one already-rendered record line, flushing a full block. */
    public synchronized void write(String recordJson) {
        if (closed) {
            throw new IllegalStateException("shard already closed: " + path);
        }
        block.append(recordJson).append('\n');
        if (++blockRecords >= BLOCK_RECORDS) {
            flushBlock();
        }
    }

    /**
     * Compress the buffered lines as one gzip member and append it.
     *
     * <p>Written in a single {@code Files.write} append so a reader never sees a
     * half-written member from a writer that is still running — only from one
     * that died.
     */
    private void flushBlock() {
        if (blockRecords == 0) {
            return;
        }
        try {
            ByteArrayOutputStream member = new ByteArrayOutputStream();
            try (OutputStream gzip = new GZIPOutputStream(member)) {
                gzip.write(block.toString().getBytes(StandardCharsets.UTF_8));
            }
            Files.write(
                    path, member.toByteArray(),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot append to " + path, e);
        } finally {
            block.setLength(0);
            blockRecords = 0;
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        flushBlock();
        closed = true;
    }
}
