package com.pricepredictor.connector.effects;

/**
 * What a collector of one game writes its records through.
 *
 * <p>The envelope fields every record of a game shares — the run, the game, the
 * attribution mode, the random seat — and the stamp applied at the write, so
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

    CollectorContext(
            RecordShardWriter writer, String gameId, AttributionMode mode,
            String randomSeatPlayerId) {
        this.writer = writer;
        this.gameId = gameId;
        this.mode = mode;
        this.randomSeatPlayerId = randomSeatPlayerId;
    }

    AttributionMode mode() {
        return mode;
    }

    String gameId() {
        return gameId;
    }

    /**
     * A record of {@code kind}, with an id and a timestamp taken now; the two
     * kinds that need more say so through {@link #newResolution} and
     * {@link #newPlayability}.
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

    /** An id for a record built later, by {@link #newResolution(Moment, String, String)}. */
    String nextRecordId() {
        return writer.nextRecordId();
    }

    /** Stamp {@code random_seat} and write. */
    void emit(EffectRecord record) {
        record.randomSeat(
                randomSeatPlayerId != null && randomSeatPlayerId.equals(record.actorPlayer()));
        writer.write(record.toJson());
    }
}
