package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * How long a command-zone object lasts, as a floating effect's or a delayed
 * trigger's {@code duration} spells it: a script's {@code Duration$} in three
 * buckets ({@link FloatingEffects#bucket}).
 *
 * <p>{@code EffectDuration} in {@code state_snapshot.py} is the other half of
 * this vocabulary, in the same order. It is not {@link EventDuration}: an
 * object's lifetime can end when something leaves play, which an event's
 * outcome has no bucket for.
 */
enum FloatingDuration implements WireNamed {
    END_OF_TURN("end_of_turn"),
    UNTIL_LEAVES("until_leaves"),
    PERMANENT("permanent");

    private final String wireName;

    FloatingDuration(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code duration} a floating effect or delayed trigger carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
