package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;

import java.util.Optional;

/**
 * 701.42b, only two meld cards can be melded: a token cannot, on scenario
 * {@code 701.42b-1}: Graf Rats' trigger exiles it and a Midnight Scavengers
 * token, and no Chittering Host enters.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, and it owns and controls a permanent whose trigger melds it, beside
 * a token it owns and controls named as its partner (see {@link Melds}). It
 * binds the token.
 *
 * <p>Variant {@code two_cards} ({@code replace_card}): the partner is the card
 * of that name rather than a token, so the two are melded.
 */
public final class TokenMeldPartner {

    public static final String RULE = "701.42b";
    static final String SCENARIO = "701.42b-1";
    static final String TWO_CARDS = "two_cards";

    private TokenMeldPartner() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TokenMeldPartner::bind, TWO_CARDS,
                new Application(ForkOperation.REPLACE_CARD, fork -> new ReplaceCardOperation(
                        fork.card("partner"), fork.card("partner").getName())));
    }

    static Optional<Binding> bind(Game live) {
        return Melds.ofActivePlayer(live).stream()
                .filter(pair -> !pair.activated() && pair.partner().isToken())
                .findFirst()
                .map(pair -> Binding.empty().with("partner", pair.partner()));
    }
}
