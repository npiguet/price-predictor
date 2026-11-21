package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * What the rules charge one candidate ({@link RulesCost#of}), computed the
 * first time it is asked and kept for the rest of one read of the
 * candidate: its verdict's {@code affordable} and its record's
 * {@code cost_after_adjustment} read the same cost. A cost that cannot be
 * computed on this board throws the same exception to every asker.
 */
final class ChargedCost {

    private final SpellAbility candidate;
    private final Player payer;
    private boolean computed;
    private RulesCost cost;
    private RuntimeException failure;

    ChargedCost(SpellAbility candidate, Player payer) {
        this.candidate = candidate;
        this.payer = payer;
    }

    /**
     * {@link RulesCost#of} for the candidate and its payer.
     *
     * @throws RuntimeException where the cost cannot be computed on this board
     */
    RulesCost get() {
        if (!computed) {
            computed = true;
            compute();
        }
        if (failure != null) {
            throw failure;
        }
        return cost;
    }

    private void compute() {
        try {
            cost = RulesCost.of(candidate, payer);
        } catch (RuntimeException e) {
            failure = e;
        }
    }
}
