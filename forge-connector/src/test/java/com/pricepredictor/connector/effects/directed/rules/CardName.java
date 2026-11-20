package com.pricepredictor.connector.effects.directed.rules;

/**
 * A card by its name in Forge's card database: the card a scenario board
 * puts in one of its places (the blocker, the attacker, the watcher).
 */
record CardName(String name) {

    /** The card named {@code name}. */
    static CardName named(String name) {
        return new CardName(name);
    }
}
