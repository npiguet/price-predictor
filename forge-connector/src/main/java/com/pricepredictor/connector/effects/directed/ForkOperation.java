package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.WireNamed;

/**
 * The seven changes a directed fork applies to a copied game (FR-011).
 *
 * <p>A scenario variant's {@code operation} in {@code rules/scenarios/*.json}
 * names one, and a directed-fork record's {@code directed.operation} carries
 * it. The reader's {@code ForkOperation} in {@code fork_operation.py} is the
 * other half of this vocabulary; {@code ForkOperationContractTest} pins both
 * spellings, so one is changed on both sides or neither.
 */
public enum ForkOperation implements WireNamed {
    /** Set a creature's base power and toughness. */
    PT("pt"),
    /** Add or remove one keyword. */
    KEYWORD("keyword"),
    /** Add or remove a colour or a type. */
    COLOUR_TYPE("colour_type"),
    /** Set a counter count. */
    COUNTERS("counters"),
    /** Tap or untap a permanent. */
    TAP("tap"),
    /** Set a player's life total. */
    LIFE("life"),
    /** Put another card in a permanent's place. */
    REPLACE_CARD("replace_card");

    private final String wireName;

    ForkOperation(String wireName) {
        this.wireName = wireName;
    }

    @Override
    public String wireName() {
        return wireName;
    }
}
