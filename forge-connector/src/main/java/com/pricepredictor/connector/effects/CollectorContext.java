package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeptRecords;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * What a collector of one game writes its records through.
 *
 * <p>The envelope fields every record of a game shares — the run, the game, the
 * attribution mode, the random seat, the rules the run aimed at — and the
 * stamp applied at the write, so
 * the three collectors build and emit a record the same way rather than each
 * keeping a copy that can drift. Each collector has its own context, because
 * each notes whether a record it wrote held the game's outcome.
 */
final class CollectorContext {

    private final RecordShardWriter writer;
    private final String gameId;
    private final AttributionMode mode;
    /**
     * The match's random seat, as a player id, or null when it has none.
     *
     * <p>Every record is stamped against it at the write (FR-030), held records
     * included: the seat is fixed for the game, so the stamp reads the same
     * whenever it is applied.
     */
    private final String randomSeatPlayerId;
    /**
     * The rules the game is aimed at (FR-009a), or null for a game aimed at
     * none: a {@code collect-coverage} or {@code collect-variants} run stamps
     * its rules, possibly none, on every record it writes, and a game a
     * steered deck plays stamps its two decks' rules instead (FR-058d,
     * {@link DeckRules}); a {@code match-outcomes} run stamps nothing, and only
     * a directed fork's records carry the rule it forked for.
     */
    private final List<String> aimedAt;
    /**
     * The flags a fork's collectors set on every record they write, before
     * the stamps above; nothing for a live game's.
     */
    private final Consumer<EffectRecord> forkFlags;
    /**
     * Where a finished record's line goes: the shard, or a fork's held lines;
     * says whether the line was taken, which the shard's cap may refuse.
     */
    private final Predicate<String> out;
    /**
     * Where an end-of-game record's line goes: the shard past the per-game
     * cap ({@link RecordShardWriter#writeUncapped}), or a fork's held lines.
     */
    private final Predicate<String> endOfGameOut;
    /** Whether this context writes a fork's records. */
    private final boolean fork;
    /**
     * The records a directed fork writes whatever its samplers draw
     * (FR-010c); nothing for a live game's or any other fork's.
     */
    private final KeptRecords kept;
    /** Whether a record this context wrote holds the game's outcome. */
    private boolean outcomeWritten;

    CollectorContext(
            RecordShardWriter writer, String gameId, AttributionMode mode,
            String randomSeatPlayerId, List<String> aimedAt) {
        this.writer = writer;
        this.gameId = gameId;
        this.mode = mode;
        this.randomSeatPlayerId = randomSeatPlayerId;
        this.aimedAt = aimedAt == null ? null : List.copyOf(aimedAt);
        this.forkFlags = record -> { };
        // Not writer::write, which would dereference a writer a collector
        // that never writes is built without.
        this.out = line -> writer.write(line);
        this.endOfGameOut = line -> {
            writer.writeUncapped(line);
            return true;
        };
        this.fork = false;
        this.kept = KeptRecords.NONE;
    }

    /** The context of a fork of {@code parent}'s game, writing through {@code output}. */
    private CollectorContext(CollectorContext parent, ForkOutput output) {
        this.writer = parent.writer;
        this.gameId = parent.gameId;
        this.mode = parent.mode;
        this.randomSeatPlayerId = parent.randomSeatPlayerId;
        this.aimedAt = parent.aimedAt;
        this.forkFlags = output.flags();
        this.out = line -> {
            output.held().accept(line);
            return true;
        };
        this.endOfGameOut = this.out;
        this.fork = true;
        this.kept = output.kept();
    }

    /**
     * The context a fork played on from this game writes through: the same
     * game id, run, mode, random seat and aim, every record first given
     * {@code output}'s flags, and each finished line handed to its held
     * lines rather than the shard, so a fork that ends in a throw can be
     * dropped whole; the records it keeps are written whatever the fork's
     * samplers draw (FR-010c).
     */
    CollectorContext forFork(ForkOutput output) {
        return new CollectorContext(this, output);
    }

    /** The records written whatever the samplers draw: none but in a directed fork. */
    KeptRecords kept() {
        return kept;
    }

    /** This game's recording has ended: the shard writer reports what its cap withheld. */
    void endGame() {
        if (writer != null) {
            writer.endGame(gameId);
        }
    }

    /** Whether the shard lost a block flush while this game was open ({@link RecordShardWriter#flushFailed}). */
    boolean flushFailed() {
        return writer != null && writer.flushFailed(gameId);
    }

    /** Whether this context writes a fork's records rather than a live game's. */
    boolean isFork() {
        return fork;
    }

    /** The records this game has had withheld by {@code --max-records-per-game} so far. */
    int withheld() {
        return writer == null ? 0 : writer.withheld(gameId);
    }

    /** Write lines a {@link #forFork fork's} context held, to this context's shard. */
    void writeHeld(List<String> lines) {
        lines.forEach(line -> writer.write(line));
    }

    AttributionMode mode() {
        return mode;
    }

    String gameId() {
        return gameId;
    }

    /**
     * A record of {@code kind}, with an id and a timestamp taken now; the
     * three kinds that need more say so through {@link #newResolution},
     * {@link #newPlayability} and {@link #newTurnAction}.
     */
    EffectRecord newRecord(RecordKind kind) {
        return new EffectRecord(
                writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                gameId, kind, mode);
    }

    /** A resolution half at {@code moment}, with an id and a timestamp taken now. */
    EffectRecord newResolution(Moment moment) {
        return newResolution(moment, writer.nextRecordId(), RecordShardWriter.timestamp());
    }

    /**
     * A resolution half at {@code moment} with an id and a timestamp taken
     * earlier.
     *
     * <p>For a record held from the moment it describes until something it
     * carries is known, such as a cast's outcome: it keeps saying when the
     * spell was cast rather than when it was written.
     */
    EffectRecord newResolution(Moment moment, String recordId, String timestamp) {
        return EffectRecord.resolution(recordId, writer.runId(), timestamp, gameId, moment, mode);
    }

    /** A playability verdict of {@code subkind}, with an id and a timestamp taken now. */
    EffectRecord newPlayability(PlayabilitySubkind subkind) {
        return EffectRecord.playability(
                writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                gameId, subkind, mode);
    }

    /** A turn-based action of {@code subkind}, with an id and a timestamp taken now. */
    EffectRecord newTurnAction(TurnActionSubkind subkind) {
        return EffectRecord.turnAction(
                writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                gameId, subkind, mode);
    }

    /** An id for a record built later, by {@link #newResolution(Moment, String, String)}. */
    String nextRecordId() {
        return writer.nextRecordId();
    }

    /**
     * {@link #emit} unless this game has already written a record saying the
     * same thing.
     *
     * <p>The one door every coalescing channel writes through, so what counts
     * as "the same thing" is {@link RecordDedup}'s single answer, read off the
     * finished record, rather than a key each channel assembles for itself and
     * can leave a field out of.
     *
     * @return whether the record reached the shard: false for a repeat, and
     *         for one the per-game cap withheld
     */
    boolean emitDistinct(EffectRecord record, RecordDedup dedup) {
        return dedup.allowDistinct(record) && emit(record);
    }

    /**
     * {@link #emitDistinct}, for a record written at the game's end: past
     * {@code --max-records-per-game} on a live game ({@link #endOfGameOut}).
     */
    boolean emitDistinctAtGameEnd(EffectRecord record, RecordDedup dedup) {
        return dedup.allowDistinct(record) && emitAtGameEnd(record);
    }

    /** {@link #emit}, for a record written at the game's end ({@link #endOfGameOut}). */
    boolean emitAtGameEnd(EffectRecord record) {
        stamp(record);
        return write(endOfGameOut, record.toJson());
    }

    /**
     * Set a fork's flags, stamp {@code random_seat} and the game's
     * {@code aimed_at}, and write.
     *
     * <p>A record already aimed keeps its own: a directed fork names the one
     * rule it forked for, which is narrower than the run's list.
     *
     * @return whether the line reached the shard: false when the per-game cap
     *         withheld it
     */
    boolean emit(EffectRecord record) {
        stamp(record);
        return write(out, record.toJson());
    }

    /**
     * Hand a finished line to {@code to}, noting whether it holds the game's
     * outcome once taken.
     *
     * @return whether {@code to} took the line
     */
    private boolean write(Predicate<String> to, String line) {
        boolean taken = to.test(line);
        outcomeWritten |= taken && RecordShardWriter.holdsTheOutcome(line);
        return taken;
    }

    /**
     * Whether a record this context wrote holds the game's outcome ({@code
     * player_lost}, {@code player_won}).
     */
    boolean wroteTheOutcome() {
        return outcomeWritten;
    }

    /** A fork's flags, {@code random_seat} and the game's {@code aimed_at}. */
    private void stamp(EffectRecord record) {
        forkFlags.accept(record);
        record.randomSeat(
                randomSeatPlayerId != null && randomSeatPlayerId.equals(record.actorPlayer()));
        if (aimedAt != null && !record.isAimed()) {
            record.aimedAt(aimedAt);
        }
    }
}
