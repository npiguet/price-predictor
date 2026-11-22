package com.pricepredictor.connector.effects;

/**
 * A record's rendered {@code state} and what its {@link RecordFacts} read of
 * it, held together from the snapshot to the write.
 *
 * @param json           the rendered state ({@link Snapshot#render})
 * @param turn           the turn its {@code global} block names, or null where
 *                       it holds none
 */
record RecordState(String json, Integer turn) {

    /** The state of a record that never read a board: {@code {}}. */
    static final RecordState EMPTY = new RecordState("{}", null);
}
