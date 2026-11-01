package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The seven record kinds, as a record's {@code kind} spells them.
 *
 * <p>The reader's {@code RecordKind} in {@code records.py} is the other half
 * of this vocabulary and refuses a kind it does not have, so a spelling is
 * changed on both sides or neither.
 */
public enum RecordKind implements WireNamed {
    /** A cost half or an effect half; which one is its {@link Moment}. */
    RESOLUTION("resolution"),
    REWRITE("rewrite"),
    CONTINUOUS("continuous"),
    COMBAT("combat"),
    TRIGGER("trigger"),
    /** A rules verdict; which question it answers is its {@link PlayabilitySubkind}. */
    PLAYABILITY("playability"),
    /**
     * A turn-based action no ability caused; which one is its {@link
     * TurnActionSubkind}. Collected only: never written to a curated stratum.
     */
    TURN_ACTION("turn_action");

    private final String wireName;

    RecordKind(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code kind} a record carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
