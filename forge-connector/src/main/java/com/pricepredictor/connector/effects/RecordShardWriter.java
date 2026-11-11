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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPOutputStream;

/**
 * One worker's append-only, gzip-compressed JSONL shard.
 *
 * <p>{@code {run_id}.{worker}-{lifetime}.jsonl.gz}, one record per line. A shard
 * per worker rather than one shared file because records are far larger than a
 * match-outcome row, so concurrent appends would interleave mid-record; and a
 * shard per <b>JVM lifetime</b> rather than per worker index because the
 * supervisor recycles the longest-running worker every status interval and
 * restarts crashed ones, so one worker index is many JVMs over a run. The two
 * counters below restart at zero with the JVM, so without a lifetime token a
 * restart reissues ids the previous JVM already used and unrelated games
 * merge under one game_id.
 *
 * <p>Compressed because the corpus is the run's real cost on disk: a game
 * produces megabytes of JSON, so an uncompressed run fills a disk long before
 * it satisfies gate 2. These records compress well — they are highly
 * repetitive JSON, the same keys and the same board described over and over.
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
 * long games, but one block is a small share of what a run collects.
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

    private final String runId;
    /** {@code {worker}-{lifetime}}: this shard's identity and the middle segment of every id it issues. */
    private final String shard;
    private final Path path;
    private final StringBuilder block = new StringBuilder();
    private int blockRecords;
    private boolean closed;
    private final AtomicLong recordCounter = new AtomicLong();
    private final AtomicLong gameCounter = new AtomicLong();
    /** The records one game may write before the rest are withheld (FR-041a). */
    private final int maxRecordsPerGame;
    /** Each game's records written and withheld, until its recording ends ({@link #endGame}). */
    private final Map<String, GameRecords> games = new HashMap<>();

    public RecordShardWriter(Path directory, String runId, int worker, String lifetime) {
        this(directory, runId, worker, lifetime, CollectionCaps.DEFAULT_MAX_RECORDS_PER_GAME);
    }

    /**
     * @param maxRecordsPerGame the records one game may write, its forks'
     *                          included; the rest are withheld and counted
     */
    public RecordShardWriter(
            Path directory, String runId, int worker, String lifetime, int maxRecordsPerGame) {
        this.maxRecordsPerGame = maxRecordsPerGame;
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

    /**
     * A file beside the shard that belongs to the same worker lifetime:
     * {@code {run_id}.{worker}-{lifetime}{suffix}}.
     */
    public Path sibling(String suffix) {
        return path.resolveSibling(runId + "." + shard + suffix);
    }

    /** Start a new game; returns its {@code game_id}. */
    public String nextGameId() {
        return runId + "." + shard + "." + gameCounter.getAndIncrement();
    }

    /** The next {@code record_id}, unique across this run's shards. */
    public String nextRecordId() {
        return runId + "." + shard + "." + recordCounter.getAndIncrement();
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

    /** ISO 8601 UTC, the timestamp format every corpus in this repo uses. */
    public static String timestamp() {
        return Instant.now().toString();
    }

    /**
     * Append one already-rendered record line, flushing a full block; or,
     * once its game has written {@code maxRecordsPerGame} records, withhold
     * it and count it against the game (FR-041a).
     */
    public synchronized void write(String recordJson) {
        if (closed) {
            throw new IllegalStateException("shard already closed: " + path);
        }
        String game = RecordLine.string(recordJson, "game_id");
        if (game != null) {
            GameRecords records = games.computeIfAbsent(game, id -> new GameRecords());
            if (records.written >= maxRecordsPerGame) {
                records.withhold(recordJson);
                return;
            }
            records.written++;
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

    /**
     * A game's recording has ended: if the cap withheld any of its records,
     * print one line naming the game, how many were withheld, and the turn
     * and acting card the most of them shared (FR-041a).
     */
    public synchronized void endGame(String gameId) {
        GameRecords records = games.remove(gameId);
        if (records != null && records.withheld > 0) {
            System.err.println(records.report(gameId, maxRecordsPerGame));
        }
    }

    /** One game's records written, and those withheld past the cap by turn and acting card. */
    private static final class GameRecords {
        private int written;
        private int withheld;
        private final Map<String, Integer> byTurnAndCard = new HashMap<>();

        void withhold(String recordJson) {
            withheld++;
            String turn = RecordLine.number(recordJson, "turn");
            String card = RecordLine.actingScript(recordJson);
            byTurnAndCard.merge("turn " + (turn == null ? "?" : turn) + ", acting card "
                    + (card == null ? "none" : card), 1, Integer::sum);
        }

        String report(String gameId, int cap) {
            // The most records; of two as many, the first by turn and card.
            Map.Entry<String, Integer> most = null;
            for (Map.Entry<String, Integer> entry : new java.util.TreeMap<>(byTurnAndCard).entrySet()) {
                if (most == null || entry.getValue() > most.getValue()) {
                    most = entry;
                }
            }
            return "Effect records: game " + gameId + " reached --max-records-per-game " + cap
                    + "; withheld " + withheld + " more records, most often (" + most.getValue()
                    + ") on " + most.getKey();
        }
    }

    /**
     * The first value of a field in a rendered record line, read without
     * parsing it: the envelope's {@code game_id}, the snapshot's
     * {@code turn}, the acting line's {@code script_file}.
     */
    static final class RecordLine {
        private RecordLine() {
        }

        /** The string value of the first {@code "key":"..."} in the line, or null. */
        static String string(String json, String key) {
            return string(json, key, 0);
        }

        /** The same, from {@code from} on. */
        private static String string(String json, String key, int from) {
            String marker = "\"" + key + "\":\"";
            int start = json.indexOf(marker, from);
            if (start < 0) {
                return null;
            }
            start += marker.length();
            int end = json.indexOf('"', start);
            return end < 0 ? null : json.substring(start, end);
        }

        /** The script file of the record's acting line, its {@code ability}'s first key, or null. */
        static String actingScript(String json) {
            int ability = json.indexOf("\"ability\":[{");
            return ability < 0 ? null : string(json, "script_file", ability);
        }

        /** The number value of the first {@code "key":N} in the line, or null. */
        static String number(String json, String key) {
            String marker = "\"" + key + "\":";
            int start = json.indexOf(marker);
            if (start < 0) {
                return null;
            }
            start += marker.length();
            int end = start;
            while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
                end++;
            }
            return end == start ? null : json.substring(start, end);
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
