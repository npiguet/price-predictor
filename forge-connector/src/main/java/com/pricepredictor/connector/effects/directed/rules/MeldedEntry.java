package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.DirectedRule;

/**
 * 712.14c, a meld card being melded with its counterpart enters the
 * battlefield as a single permanent with its back face up, on scenario
 * {@code 712.4a-2} (shared with 712.4a): Graf Rats beside Hill Giant melds
 * nothing; beside Midnight Scavengers the two would return as Chittering
 * Host.
 *
 * <p>The base board is 701.42a's ({@link Meld}): it binds the host and a
 * creature its partner can take the place of.
 *
 * <p>Variant {@code meld_pair} ({@code replace_card}): the partner is in that
 * creature's place, so the pair melds and enters combined.
 */
public final class MeldedEntry {

    public static final String RULE = "712.14c";
    static final String SCENARIO = "712.4a-2";
    static final String MELD_PAIR = "meld_pair";

    private MeldedEntry() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, MELD_PAIR, Meld.pairing());
    }
}
