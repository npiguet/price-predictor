package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * How long an event's outcome lasts, as its {@code duration} spells it.
 *
 * <p>The effect head's three buckets; {@code DURATIONS} in
 * {@code effect_head_layout.py} is the other half of this vocabulary. Forge
 * spells a script's duration a dozen ways, and {@link EventAttribution} is
 * what folds them into these.
 */
public enum EventDuration implements WireNamed {
    INSTANT("instant"),
    END_OF_TURN("end_of_turn"),
    PERMANENT("permanent");

    private final String wireName;

    EventDuration(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code duration} an event carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
