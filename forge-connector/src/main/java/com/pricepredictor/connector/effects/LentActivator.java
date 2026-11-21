package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.function.BooleanSupplier;

/**
 * A rules read of a candidate nobody has activated, which has no activating
 * player and which Forge's target and cost readers dereference: the random
 * seat's draw gates read each candidate through here.
 *
 * <p>The read runs in the candidate's own {@link ActivatorScope}, with no
 * scan of the game's other abilities: it sets nothing on them. The
 * activating player and the choices the read finds are put back after it,
 * also when it throws, as the verdict's are ({@link PlayabilityVerdict#asActivator}).
 */
public final class LentActivator {

    private LentActivator() {
    }

    /**
     * {@code read} with {@code activator} lent to {@code candidate}. A read
     * that throws on this board is counted and reads as false, as a verdict
     * whose check throws reads as unplayable ({@link DecisionCandidate#read}).
     */
    public static boolean holds(SpellAbility candidate, Player activator, BooleanSupplier read) {
        try {
            return ActivatorScope.alone(candidate, activator).run(read::getAsBoolean);
        } catch (RuntimeException e) {
            HookFailures.countContained(new HookFailures.FailureKey("random seat gate unreadable"), e);
            return false;
        }
    }
}
