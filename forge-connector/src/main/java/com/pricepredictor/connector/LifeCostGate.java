package com.pricepredictor.connector;

import forge.game.cost.Cost;
import forge.game.cost.CostPart;
import forge.game.cost.CostPayLife;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * The random seat's one refusal on a cost: it never draws a play whose life
 * payment would leave it at 0 or less.
 *
 * <p>The rules let a player pay life down to 0 ({@code canPayLife} is true
 * at 1 life) and the verdict still calls such a play playable; only the
 * seat's draw leaves it out, so a game does not end on the seat's own
 * payment.
 */
final class LifeCostGate {

    private LifeCostGate() {
    }

    /** Whether paying {@code sa}'s life costs would leave {@code payer} at 0 life or less. */
    static boolean wouldLeaveNoLife(SpellAbility sa, Player payer) {
        Cost cost = sa.getPayCosts();
        if (cost == null) {
            return false;
        }
        int total = 0;
        for (CostPart part : cost.getCostParts()) {
            if (part instanceof CostPayLife life) {
                total += life.getAbilityAmount(sa);
            }
        }
        return total > 0 && total >= payer.getLife();
    }
}
