package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;

/**
 * 712.4a, one card of a meld pair exiles both and melds them, putting them
 * onto the battlefield combined with their back faces up, on scenario
 * {@code 712.4a-1}: at the beginning of combat Graf Rats' trigger melds it
 * with Midnight Scavengers into Chittering Host.
 *
 * <p>The base board is 701.42a's ({@link Meld}): the active player's first
 * main phase begins, and it owns and controls a permanent whose ability melds
 * it beside a card named as its partner. It binds the partner.
 *
 * <p>Variant {@code wrong_partner} ({@code replace_card}): Hill Giant is in
 * the partner's place, so nothing is exiled and no melded permanent enters.
 */
public final class MeldPair {

    public static final String RULE = "712.4a";
    static final String SCENARIO = "712.4a-1";
    static final String WRONG_PARTNER = "wrong_partner";

    private MeldPair() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, WRONG_PARTNER,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("partner"), Meld.GIANT)));
    }
}
