package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;

import java.util.Optional;

/**
 * 701.42c, objects that cannot be melded are not, on scenario
 * {@code 701.42c-1}: Hanweir Battlements' meld ability exiles it and a Hanweir
 * Garrison token, and no Hanweir, the Writhing Township enters.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, and it owns and controls a permanent whose activated ability, which
 * it has the lands for, melds it, beside a token it owns and controls named
 * as its partner (see {@link Melds}). It binds the token.
 *
 * <p>Variant {@code real_garrison} ({@code replace_card}): the partner is the
 * card of that name rather than a token, so the two are melded.
 */
public final class UnmeldableStay {

    public static final String RULE = "701.42c";
    static final String SCENARIO = "701.42c-1";
    static final String REAL_PARTNER = "real_garrison";

    private UnmeldableStay() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, UnmeldableStay::bind, REAL_PARTNER,
                new Application(ForkOperation.REPLACE_CARD, fork -> new ReplaceCardOperation(
                        fork.card("partner"), fork.card("partner").getName())));
    }

    static Optional<Binding> bind(Game live) {
        return Melds.ofActivePlayer(live).stream()
                .filter(pair -> pair.activated() && pair.partner().isToken())
                .findFirst()
                .map(pair -> Binding.empty().with("partner", pair.partner()));
    }
}
