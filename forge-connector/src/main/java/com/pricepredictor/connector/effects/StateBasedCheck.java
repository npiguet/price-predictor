package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventCardStatsChanged;

import java.util.HashSet;
import java.util.Set;

/**
 * A fork's state-based check, run as the live priority check runs it: the
 * stat changes it made are published after its loop (Forge's {@code
 * PhaseHandler.checkStateBasedEffectsUnobserved}), which {@code
 * GameAction.checkStateEffects(false, ...)} leaves to its caller.
 *
 * <p>Damage alone kills nothing; a creature with lethal damage on it leaves
 * the battlefield during state-based actions, and a probe branch stopping
 * before them would miss every death the real step had, biasing the
 * difference gate 2 computes towards "the keyword changed nothing". The stat
 * changes those deaths cause (a lord dying, the creatures it pumped
 * shrinking), or an Aura or Equipment falling off a creature a fork's edit
 * made protected, reach the records through the stats event; a fork that did
 * not publish it would read them as its perturbation's doing, or lose them.
 */
final class StateBasedCheck {

    private StateBasedCheck() {
    }

    /** Let the deaths and the falls happen, and publish the stat changes they made. */
    static void run(Game fork) {
        Set<Card> affected = new HashSet<>();
        fork.getAction().checkStateEffects(false, affected);
        if (!affected.isEmpty()) {
            fork.fireEvent(new GameEventCardStatsChanged(affected));
        }
    }
}
