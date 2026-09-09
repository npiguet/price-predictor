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
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Pattern;
import java.util.zip.GZIPOutputStream;

/**
 * One worker's append-only, gzip-compressed JSONL shard.
 *
 * <p>{@code {run_id}.{worker}-{lifetime}.jsonl.gz}, one record per line. A shard
 * per worker rather than one shared file because records are far larger than a
 * match-outcome row, so concurrent appends would interleave mid-record; and a
 * shard per <b>JVM lifetime</b> rather than per worker index because the
 * supervisor recycles the longest-running worker every status interval and
 * restarts crashed ones — roughly 530 JVMs over an eight-hour run. The two
 * counters below restart at zero with the JVM, so without a lifetime token a
 * restart reissues ids the previous JVM already used and dozens of unrelated
 * games merge under one game_id.
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
 * <p>Owns the two counters the record ids are built from. Both carry the shard
 * token — worker index and lifetime — because workers count independently and
 * each JVM counts from zero.
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

    /** A lifetime token's alphabet: no dot and no hyphen, so an id still splits cleanly. */
    private static final Pattern LIFETIME = Pattern.compile("[0-9a-z]{1,16}");

    private final String runId;
    /** {@code {worker}-{lifetime}}: this shard's identity and the middle segment of every id it issues. */
    private final String shard;
    private final Path path;
    private final StringBuilder block = new StringBuilder();
    private int blockRecords;
    private boolean closed;
    private long recordCounter;
    private long gameCounter;

    public RecordShardWriter(Path directory, String runId, int worker, String lifetime) {
        this.runId = runId;
        this.shard = worker + "-" + lifetime;
        this.path = directory.resolve(runId + "." + shard + SUFFIX);
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
        return runId + "." + shard + "." + gameCounter++;
    }

    /** The next {@code record_id}, unique across this run's shards. */
    public String nextRecordId() {
        return runId + "." + shard + "." + recordCounter++;
    }

    public String runId() {
        return runId;
    }

    /**
     * A token unique to this JVM lifetime.
     *
     * <p>Base-36 start millis, so a directory listing is in start order, plus
     * four random characters, so two JVMs that start in the same millisecond
     * under one run id and worker index still differ. Minted here rather than
     * handed down by the supervisor because several launchers spawn this worker
     * and one that forgot to pass a token would silently reissue a previous
     * JVM's ids.
     */
    public static String mintLifetime() {
        return Long.toString(System.currentTimeMillis(), 36)
                + Integer.toString(
                        ThreadLocalRandom.current()
                                .nextInt(36 * 36 * 36, 36 * 36 * 36 * 36), 36);
    }

    /** Whether {@code token} is safe to embed in a shard name and an id. */
    public static boolean isValidLifetime(String token) {
        return token != null && LIFETIME.matcher(token).matches();
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
