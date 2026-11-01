package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * Which turn-based action a {@code turn_action} record holds (FR-036).
 *
 * <p>Every turn-action record has one and no other kind does, which is why a
 * record is given its subkind when it is made, by
 * {@link EffectRecord#turnAction}. {@code TurnActionSubkind} in
 * {@code records.py} is the other half of this vocabulary.
 */
public enum TurnActionSubkind implements WireNamed {
    /** The active player's draw for the turn (CR 504.1). */
    DRAW("draw"),
    /** The discard to maximum hand size (CR 514.1). */
    CLEANUP_DISCARD("cleanup_discard"),
    /** The active player's permanents untapping (CR 502.3). */
    UNTAP("untap"),
    /** Permanents phasing in and out (CR 502.1). */
    PHASING("phasing"),
    /** Day becoming night or night day (CR 502.2). */
    DAY_NIGHT("day_night"),
    /** A lore counter on each of the active player's Sagas (CR 714.3b). */
    LORE_COUNTER("lore_counter");

    private final String wireName;

    TurnActionSubkind(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code subkind} a record carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
