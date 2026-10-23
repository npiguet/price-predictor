package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;
import forge.game.zone.MagicStack;

import java.lang.reflect.Field;

/**
 * The player action {@code MagicStack} has in progress, which the stack keeps
 * to itself.
 *
 * <p>Every action a player takes at priority — a cast, an activation, a
 * special action — goes through {@code MagicStack.freezeStack(sa)}, which
 * names the action as the stack's primary ability until {@code
 * addAndUnfreeze} lets it go. Nothing else names it: a state-based check
 * freezes the stack too ({@code GameAction.checkStateEffects} calls {@code
 * setFrozen(true)}) but names no ability, and a resolution freezes it with
 * none. So the primary ability is what tells a special action — which
 * {@code MagicStack.add} resolves in place, with no push and no cast event —
 * from the state-based check that follows a cast.
 *
 * <p>An ability the AI gave up on is no action in progress, though the stack
 * may still name it: {@code ComputerUtil.handlePlayingSpellAbility} returns on
 * a failed payment without unfreezing, and marks the ability skipped.
 *
 * <p>Read reflectively, as Forge publishes no accessor. On a checkout where
 * the field cannot be read, no action reads as in progress, what a special
 * action does is filed as the state-based check would be, and stderr says so
 * once.
 */
final class StackActions {

    private static final Field PRIMARY_ABILITY = lookup();

    private StackActions() {
    }

    private static Field lookup() {
        try {
            Field field = MagicStack.class.getDeclaredField("primaryAbility");
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException e) {
            System.err.println("effect records: MagicStack.primaryAbility cannot be read ("
                    + e + "); a special action at priority will be filed as the"
                    + " state-based check after the last cast");
            return null;
        }
    }

    /** The ability whose action is in progress on {@code stack}, or null. */
    static SpellAbility inProgress(MagicStack stack) {
        if (PRIMARY_ABILITY == null || stack == null) {
            return null;
        }
        try {
            SpellAbility primary = (SpellAbility) PRIMARY_ABILITY.get(stack);
            return primary == null || primary.isSkip() ? null : primary;
        } catch (IllegalAccessException | ClassCastException e) {
            return null;
        }
    }
}
