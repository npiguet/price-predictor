package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;

/**
 * 712.14c, a meld card being melded with its counterpart enters the
 * battlefield as a single permanent with its back face up, on scenario
 * {@code 712.4a-1} (shared with 712.4a): Graf Rats and Midnight Scavengers
 * return as Chittering Host.
 *
 * <p>The base board is 701.42a's ({@link Meld}): the active player's first
 * main phase begins, and it owns and controls a permanent whose ability melds
 * it beside a card named as its partner. It binds the partner.
 *
 * <p>Variant {@code wrong_partner} ({@code replace_card}): Hill Giant is in
 * the partner's place, so no pair is melded and nothing enters combined.
 */
public final class MeldedEntry {

    public static final String RULE = "712.14c";
    static final String SCENARIO = "712.4a-1";
    static final String WRONG_PARTNER = "wrong_partner";

    private MeldedEntry() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, WRONG_PARTNER,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("partner"), Meld.GIANT)));
    }
}
