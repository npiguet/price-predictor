package com.pricepredictor.connector.effects;

/**
 * What the shard writer's per-game cap reads of a record (FR-041a), handed
 * to it beside the rendered line by the record's maker, which holds them.
 *
 * @param gameId       the record's {@code game_id}
 * @param turn         the turn its state was read on, or null for a record
 *                     whose state holds no {@code global} block
 * @param actingScript the script file of its acting line, or of the line its
 *                     payload names when it names none of its own (a
 *                     {@code decision} record's candidate); null for neither
 * @param holdsOutcome whether it is a live record holding the game's outcome
 *                     ({@code player_lost} or {@code player_won}), which the
 *                     cap never withholds
 */
public record RecordFacts(String gameId, Integer turn, String actingScript, boolean holdsOutcome) {
}
