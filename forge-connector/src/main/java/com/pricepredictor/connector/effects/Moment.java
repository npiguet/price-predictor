package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * Which half of a resolution pair a {@code resolution} record is.
 *
 * <p>Every resolution record has one and no other kind does, which is why a
 * record is given its moment when it is made, by {@link EffectRecord#resolution}.
 * {@code Moment} in {@code records.py} is the other half of this vocabulary.
 */
public enum Moment implements WireNamed {
    /** The cost half: what was paid, and what became of the spell. */
    ACTIVATION("activation"),
    /** The effect half: what the resolution did. */
    RESOLUTION("resolution");

    private final String wireName;

    Moment(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code moment} a record carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
