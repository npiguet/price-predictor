package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * Why an entity beyond the four inclusion tiers is in the snapshot, as its
 * {@code role} spells it.
 *
 * <p>{@code EntityRole} in {@code state_snapshot.py} is the other half of this
 * vocabulary, in the same order.
 */
enum EntityRole implements WireNamed {
    /** A library's top card, or a library card an event names. */
    LIBRARY_CARD("library_card"),
    /** A card exiled with a permanent of the tiers. */
    EXILED_WITH("exiled_with"),
    /** A card merged into a mutated permanent. */
    MUTATE_COMPONENT("mutate_component"),
    /** The pending event's subject as it last existed, under id {@code L<card id>}. */
    LAST_KNOWN("last_known");

    private final String wireName;

    EntityRole(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code role} an entity carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
