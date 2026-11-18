package com.pricepredictor.connector.ability;

import forge.game.cost.Cost;
import forge.game.keyword.KeywordInterface;

/**
 * A Kicker keyword's second cost, read from {@code Kicker:<cost>[:<second cost>]}:
 * a card with two kicker costs prints them joined by "and/or".
 */
record Kicker(String secondCost) {

    static Kicker of(KeywordInterface ki) {
        return new Kicker(new KeywordSpec(ki.getOriginal()).rest(2));
    }

    /** {@code title} followed by the second cost, when the keyword has one. */
    String titled(String title) {
        return secondCost == null
                ? title : title + " and/or " + new Cost(secondCost, false).toSimpleString();
    }
}
