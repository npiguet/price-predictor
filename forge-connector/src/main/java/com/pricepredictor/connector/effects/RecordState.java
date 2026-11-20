package com.pricepredictor.connector.effects;

/**
 * A record's rendered {@code state} and what its {@link RecordFacts} read of
 * it, held together from the snapshot to the write.
 *
 * @param json           the rendered state ({@link Snapshot#render})
 * @param turn           the turn its {@code global} block names, or null where
 *                       it holds none
 * @param pendingOutcome whether its {@code pending_event} is the game's
 *                       outcome ({@code player_lost}, {@code player_won})
 */
record RecordState(String json, Integer turn, boolean pendingOutcome) {

    /** The state of a record that never read a board: {@code {}}. */
    static final RecordState EMPTY = new RecordState("{}", null, false);
}
