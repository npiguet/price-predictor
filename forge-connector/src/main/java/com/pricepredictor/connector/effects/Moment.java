package com.pricepredictor.connector.effects;

/**
 * Which half of a resolution pair a {@code resolution} record is.
 *
 * <p>Every resolution record has one and no other kind does, which is why a
 * record is given its moment when it is made, by {@link EffectRecord#resolution},
 * rather than by a setter a caller could forget.
 */
public enum Moment {
    /** The cost half: what was paid, and what became of the spell. */
    ACTIVATION("activation"),
    /** The effect half: what the resolution did. */
    RESOLUTION("resolution");

    private final String wireName;

    Moment(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code moment} a record carries. */
    public String wireName() {
        return wireName;
    }
}
