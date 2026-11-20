package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The game's day or night, as the global state's {@code day_night} spells it;
 * null when the game is neither.
 *
 * <p>{@code DayNight} in {@code state_snapshot.py} is the other half of this
 * vocabulary, in the same order.
 */
enum DayNight implements WireNamed {
    DAY("day"),
    NIGHT("night");

    private final String wireName;

    DayNight(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code day_night} a snapshot's global state carries. */
    @Override
    public String wireName() {
        return wireName;
    }

    /** The side Forge's daytime flag names: true at night, null for neither. */
    static DayNight of(Boolean night) {
        return night == null ? null : night ? NIGHT : DAY;
    }
}
