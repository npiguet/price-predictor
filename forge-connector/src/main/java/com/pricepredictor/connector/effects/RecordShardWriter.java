package com.pricepredictor.connector.effects;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
 * <p>A member also ends at every game's end ({@link #endGame}), after the
 * game's game-end marker: a finished game is on disk whole before the next
 * one starts, so a kill or a crash, which runs no shutdown hook, loses only
 * the game in progress, never the tail and marker of the game before. The
 * blocks are written in record order, so a marker is never on disk while an
 * earlier record of its game is lost. The cost is a member per game, a
 * fraction of a percent of a shard's size.
 *
 * <p>A crash still loses the game in progress: its unwritten records,
 * at most {@code BLOCK_RECORDS - 1}, and its marker, so the game reads as cut
 * whole.
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

    /**
     * Consecutive failed appends after which the writer fails for good. One
     * failure followed by a success is a hiccup the kept block rides out; this
     * many in a row is a full disk or a lost share, and the kept block would
     * only grow in the worker's heap. The block then holds at most
     * {@code BLOCK_RECORDS + MAX_FAILED_APPENDS - 1} records.
     */
    static final int MAX_FAILED_APPENDS = 3;

    /**
     * The writer can no longer append and has dropped what it held. An
     * {@link Error}, so it passes the game's, the match's and the worker
     * loop's {@code catch (Exception)} (and the engine's hook containment,
     * which lets an {@code Error} through) and ends the worker, which the
     * supervisor reports as a crash.
     */
    public static final class ShardUnwritable extends Error {
        ShardUnwritable(String message, IOException last) {
            super(message, last);
        }
    }

    private final String runId;
    /** {@code {worker}-{lifetime}}: this shard's identity and the middle segment of every id it issues. */
    private final String shard;
    private final Path path;
    private final ShardAppender appender;
    private final StringBuilder block = new StringBuilder();
    private int blockRecords;
    private boolean closed;
    private int failedAppends;
    private ShardUnwritable gaveUp;
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
        this.appender = new ShardAppender(path);
        try {
            Files.createDirectories(directory);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot open effect-record shard " + path, e);
        }
    }

    /** {@code {worker}-{lifetime}}: the token that names this writer's worker lifetime. */
    public String token() {
        return shard;
    }

    /** Append through {@code channel}: a test's, which fails. */
    void appendThrough(ShardAppender.Channel channel) {
        appender.through(channel);
    }

    /** The records buffered and not yet appended. */
    synchronized int heldRecords() {
        return blockRecords;
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
     *
     * <p>A live record holding the game's outcome ({@link
     * RecordFacts#holdsOutcome}) is written past the cap too, or a capped
     * game would keep its marker and lose its result. A fork's record
     * holding them holds the fork's outcome, not the game's: the cap
     * withholds it as any other record, rather than write it while the live
     * record it mirrors is withheld.
     *
     * @param facts what the record's maker held of it: its game, turn,
     *              acting script and whether it holds the game's outcome
     * @return whether the line was written, false for one the cap withheld
     */
    public synchronized boolean write(String recordJson, RecordFacts facts) {
        requireWritable();
        if (facts.gameId() != null && !games.computeIfAbsent(facts.gameId(), id -> new GameRecords())
                .admit(facts, maxRecordsPerGame)) {
            return false;
        }
        block.append(recordJson).append('\n');
        if (++blockRecords >= BLOCK_RECORDS) {
            flushBlock();
        }
        return true;
    }

    /** Refuses a write to a closed shard, or to one that failed for good. */
    private void requireWritable() {
        if (closed) {
            throw new IllegalStateException("shard already closed: " + path);
        }
        if (gaveUp != null) {
            throw gaveUp;
        }
    }

    /**
     * Append one record line past {@code --max-records-per-game}: a game's
     * end-of-game records (the reservoirs' trigger and mana records, the
     * game-end marker), each kept by a per-key bound of its own, are written
     * last and so would be the first a cap withheld (V3 #5).
     */
    public synchronized void writeUncapped(String recordJson) {
        requireWritable();
        block.append(recordJson).append('\n');
        if (++blockRecords >= BLOCK_RECORDS) {
            flushBlock();
        }
    }

    /** The records {@code gameId} has had withheld so far; 0 for a game that had none. */
    public synchronized int withheld(String gameId) {
        GameRecords records = games.get(gameId);
        return records == null ? 0 : records.withheld;
    }

    /**
     * Compress the buffered lines as one gzip member and append it.
     *
     * <p>Written in a single append so a reader never sees a half-written
     * member from a writer that is still running — only from one that died.
     * An append that fails is cut back ({@link ShardAppender}) and the block
     * is kept for the next flush; every game open now is marked
     * ({@link #flushFailed}) so it gets no game-end marker.
     */
    private void flushBlock() {
        if (gaveUp != null) {
            throw gaveUp;
        }
        if (blockRecords == 0) {
            return;
        }
        try {
            ByteArrayOutputStream member = new ByteArrayOutputStream();
            try (OutputStream gzip = new GZIPOutputStream(member)) {
                gzip.write(block.toString().getBytes(StandardCharsets.UTF_8));
            }
            appender.append(member.toByteArray());
        } catch (IOException e) {
            games.values().forEach(game -> game.flushFailed = true);
            if (++failedAppends >= MAX_FAILED_APPENDS) {
                block.setLength(0);
                blockRecords = 0;
                gaveUp = new ShardUnwritable("effect-record shard " + path + " could not be"
                        + " appended to " + failedAppends + " times in a row; last failure: " + e, e);
                throw gaveUp;
            }
            throw new UncheckedIOException("cannot append to " + path, e);
        }
        failedAppends = 0;
        block.setLength(0);
        blockRecords = 0;
    }

    /**
     * Whether a block flush failed while {@code gameId} was open: its
     * records were held back past their write, so it is not marked as ended
     * whole.
     */
    public synchronized boolean flushFailed(String gameId) {
        GameRecords records = games.get(gameId);
        return records != null && records.flushFailed;
    }

    /**
     * A game's recording has ended, its marker written last: the block in
     * flight is written, so the game is on disk whole before the next one
     * starts; and if the cap withheld any of its records, print one line
     * naming the game, how many were withheld, and the turn and acting card
     * the most of them shared (FR-041a).
     */
    public synchronized void endGame(String gameId) {
        try {
            if (!closed) {
                flushBlock();
            }
        } finally {
            GameRecords records = games.remove(gameId);
            if (records != null && records.withheld > 0) {
                System.err.println(records.report(gameId, maxRecordsPerGame));
            }
        }
    }

    /** One game's records written, and those withheld past the cap by turn and acting card. */
    private static final class GameRecords {
        private int written;
        private int withheld;
        private boolean flushFailed;
        private final Map<String, Integer> byTurnAndCard = new HashMap<>();

        /**
         * Count a record against the game: written while under {@code cap},
         * or holding the game's outcome; else withheld.
         *
         * @return whether it is written
         */
        boolean admit(RecordFacts facts, int cap) {
            if (written >= cap && !facts.holdsOutcome()) {
                withhold(facts);
                return false;
            }
            written++;
            return true;
        }

        private void withhold(RecordFacts facts) {
            withheld++;
            Integer turn = facts.turn();
            String card = facts.actingScript();
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

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        try {
            if (gaveUp == null) {
                flushBlock();
            }
        } finally {
            closed = true;
        }
    }
}
