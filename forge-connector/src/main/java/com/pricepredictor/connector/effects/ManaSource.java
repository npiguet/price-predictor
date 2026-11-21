package com.pricepredictor.connector.effects;

import java.util.List;

/**
 * One card a player could pay a candidate's mana cost with, as
 * {@link ManaAffordability} searches it: its choices of what it pays, the
 * mana each of its mana abilities would make and the unit it pays as a
 * keyword's tapper. A payment takes one choice per source.
 *
 * @param options each choice's units, in the order the search tries them;
 *                never empty
 */
record ManaSource(List<List<ManaUnit>> options) {

    /** How many choices the source offers. */
    int choices() {
        return options.size();
    }

    /** The source with its first choice alone. */
    ManaSource firstOnly() {
        return new ManaSource(List.of(options.get(0)));
    }
}
