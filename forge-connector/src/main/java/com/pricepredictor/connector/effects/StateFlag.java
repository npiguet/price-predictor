package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The flags a {@code state_flag_change} event's {@code flag} param names, in
 * the order of {@code _STATE_FLAG_FIELDS} in {@code effect_targets.py} with
 * {@link #CLASS_LEVEL} first; {@code test_wire_closed_lists.py} pins the two
 * against each other. {@link #CLASS_LEVEL} carries the level reached as its
 * value, every other flag a {@code true}.
 */
public enum StateFlag implements WireNamed {
    /** A Class enchantment gained a level; the value is the level reached. */
    CLASS_LEVEL("class_level"),
    /** A Room's door was unlocked. */
    DOOR_UNLOCKED("door_unlocked"),
    /** A creature became paired (soulbond). */
    PAIRED("paired"),
    /** A permanent merged into a mutated one. */
    MERGED("merged"),
    /** A creature joined the attackers after attackers were declared. */
    ENTERED_ATTACKING("entered_attacking"),
    /** A creature joined the blockers after blockers were declared. */
    ENTERED_BLOCKING("entered_blocking"),
    /** A player got the city's blessing (ascend). */
    CITY_BLESSING("city_blessing"),
    /** A player completed a dungeon. */
    DUNGEON_COMPLETED("dungeon_completed");

    private final String wireName;

    StateFlag(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code flag} the event carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
