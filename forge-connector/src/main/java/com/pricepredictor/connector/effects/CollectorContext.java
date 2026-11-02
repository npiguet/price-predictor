package com.pricepredictor.connector.effects;

import java.util.List;
import java.util.function.Consumer;

/**
 * What a collector of one game writes its records through.
 *
 * <p>The envelope fields every record of a game shares — the run, the game, the
 * attribution mode, the random seat, the rules the run aimed at — and the
 * stamp applied at the write, so
 * the three collectors build and emit a record the same way rather than each
 * keeping a copy that can drift. Each collector has its own context: they
 * count their records separately.
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
     * The rules the run aimed at (FR-009a), or null for a run that aimed at
     * none: a {@code collect-coverage} or {@code collect-variants} run stamps
     * them, possibly none, on every record it writes; a {@code match-outcomes}
     * run stamps nothing, and only a directed fork's records carry the rule it
     * forked for.
     */
    private final List<String> aimedAt;
    /**
     * The flags a fork's collectors set on every record they write, before
     * the stamps above; nothing for a live game's.
     */
    private final Consumer<EffectRecord> forkFlags;
    /** Where a finished record's line goes: the shard, or a fork's held lines. */
    private final Consumer<String> out;

    CollectorContext(
            RecordShardWriter writer, String gameId, AttributionMode mode,
            String randomSeatPlayerId, List<String> aimedAt) {
        this(writer, gameId, mode, randomSeatPlayerId,
                aimedAt == null ? null : List.copyOf(aimedAt), record -> { },
                // Not writer::write, which would dereference a writer a
                // collector that never writes is built without.
                line -> writer.write(line));
    }

    private CollectorContext(
            RecordShardWriter writer, String gameId, AttributionMode mode,
            String randomSeatPlayerId, List<String> aimedAt,
            Consumer<EffectRecord> forkFlags, Consumer<String> out) {
        this.writer = writer;
        this.gameId = gameId;
        this.mode = mode;
        this.randomSeatPlayerId = randomSeatPlayerId;
        this.aimedAt = aimedAt;
        this.forkFlags = forkFlags;
        this.out = out;
    }

    /**
     * The context a fork played on from this game writes through: the same
     * game id, run, mode, random seat and aim, every record first given
     * {@code flags}, and each finished line handed to {@code out} rather
     * than the shard, so a fork that ends in a throw can be dropped whole.
     */
    CollectorContext forFork(Consumer<EffectRecord> flags, Consumer<String> out) {
        return new CollectorContext(writer, gameId, mode, randomSeatPlayerId, aimedAt, flags, out);
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
     * @return whether the record was written
     */
    boolean emitDistinct(EffectRecord record, RecordDedup dedup) {
        if (!dedup.allowDistinct(record)) {
            return false;
        }
        emit(record);
        return true;
    }

    /**
     * Set a fork's flags, stamp {@code random_seat} and the run's
     * {@code aimed_at}, and write.
     *
     * <p>A record already aimed keeps its own: a directed fork names the one
     * rule it forked for, which is narrower than the run's list.
     */
    void emit(EffectRecord record) {
        forkFlags.accept(record);
        record.randomSeat(
                randomSeatPlayerId != null && randomSeatPlayerId.equals(record.actorPlayer()));
        if (aimedAt != null && !record.isAimed()) {
            record.aimedAt(aimedAt);
        }
        out.accept(record.toJson());
    }
}
