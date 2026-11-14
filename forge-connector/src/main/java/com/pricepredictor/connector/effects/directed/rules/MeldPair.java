package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.DirectedRule;

/**
 * 712.4a, a meld card's front face and its counterpart meld into one
 * permanent, on scenario {@code 712.4a-2} (shared with 712.14c), 712.4a-1
 * inverted: Graf Rats beside Hill Giant melds nothing; beside Midnight
 * Scavengers both would be exiled and return as Chittering Host.
 *
 * <p>The base board is 701.42a's ({@link Meld}): it binds the host and a
 * creature its partner can take the place of.
 *
 * <p>Variant {@code meld_pair} ({@code replace_card}): the partner is in that
 * creature's place, so the pair is exiled and melded.
 */
public final class MeldPair {

    public static final String RULE = "712.4a";
    static final String SCENARIO = "712.4a-2";
    static final String MELD_PAIR = "meld_pair";

    private MeldPair() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, MELD_PAIR, Meld.pairing());
    }
}
