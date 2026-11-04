package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;

import java.util.Optional;

/**
 * 701.42a, melding exiles a meld pair and puts them onto the battlefield as
 * one permanent, on scenario {@code 701.42a-1}: at the beginning of combat
 * Graf Rats' trigger melds it with Midnight Scavengers into Chittering Host.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, and it owns and controls a permanent whose trigger or affordable
 * activated ability melds it, beside a card (not a token) it owns and
 * controls named as its partner (see {@link Melds}). It binds the partner.
 *
 * <p>Variant {@code wrong_partner} ({@code replace_card}): Hill Giant is in
 * the partner's place, so there is nothing to meld with and nothing is exiled.
 */
public final class Meld {

    public static final String RULE = "701.42a";
    static final String SCENARIO = "701.42a-1";
    static final String WRONG_PARTNER = "wrong_partner";
    static final String GIANT = "Hill Giant";

    private Meld() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, WRONG_PARTNER,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("partner"), GIANT)));
    }

    static Optional<Binding> bind(Game live) {
        return Melds.ofActivePlayer(live).stream()
                .filter(pair -> !pair.partner().isToken())
                .findFirst()
                .map(pair -> Binding.empty().with("partner", pair.partner()));
    }
}
