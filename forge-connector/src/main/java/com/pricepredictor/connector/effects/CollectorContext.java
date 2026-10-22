package com.pricepredictor.connector.effects;

/**
 * What a collector of one game writes its records through.
 *
 * <p>The envelope fields every record of a game shares — the run, the game, the
 * attribution mode — and the stamp applied at the write, so the three
 * collectors build and emit a record the same way rather than each keeping a
 * copy that can drift. Each collector has its own context: they count their
 * records separately.
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
     * whenever it is applied. Set after the game is created and before the
     * first event, because the seat is decided per match and the collector per
     * game.
     */
    private String randomSeatPlayerId;
    private long recordsWritten;

    CollectorContext(RecordShardWriter writer, String gameId) {
        this.writer = writer;
        this.gameId = gameId;
        this.mode = AttributionMode.detect();
    }

    AttributionMode mode() {
        return mode;
    }

    String gameId() {
        return gameId;
    }

    /** A record of {@code kind}, with an id and a timestamp taken now. */
    EffectRecord newRecord(String kind) {
        return newRecord(kind, writer.nextRecordId(), RecordShardWriter.timestamp());
    }

    /**
     * A record of {@code kind} with an id and a timestamp taken earlier.
     *
     * <p>For a record held from the moment it describes until something it
     * carries is known, such as a cast's outcome: it keeps saying when the
     * spell was cast rather than when it was written.
     */
    EffectRecord newRecord(String kind, String recordId, String timestamp) {
        return new EffectRecord(recordId, writer.runId(), timestamp, gameId, kind, mode);
    }

    /** An id for a record that will be built later, from {@link #newRecord(String, String, String)}. */
    String nextRecordId() {
        return writer.nextRecordId();
    }

    String randomSeatPlayerId() {
        return randomSeatPlayerId;
    }

    void setRandomSeatPlayerId(String playerId) {
        this.randomSeatPlayerId = playerId;
    }

    /** Stamp {@code random_seat} and write. */
    void emit(EffectRecord record) {
        record.randomSeat(
                randomSeatPlayerId != null && randomSeatPlayerId.equals(record.actorPlayer()));
        writer.write(record.toJson());
        recordsWritten++;
    }

    long recordsWritten() {
        return recordsWritten;
    }
}
