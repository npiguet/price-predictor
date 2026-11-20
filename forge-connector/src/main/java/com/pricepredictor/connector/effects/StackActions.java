package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;
import forge.game.zone.MagicStack;

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
 * <p>A failed payment leaves the stack naming an ability nobody is playing:
 * {@code ComputerUtil.handlePlayingSpellAbility} returns without unfreezing,
 * and marks the ability skipped, which this reads as no action in progress.
 * That holds only until {@code AiController.getSpellAbilityToPlay} clears the
 * skip flags of the last choice, on the AI's next look for something to
 * play. From then until the stack is next unfrozen — the next resolution, a
 * land played, a step — the stale primary reads as an action in progress, so
 * what is filed then goes to no record and ends the last cast's
 * consequences. The impact is small: a frozen stack parks every cast and
 * trigger added to it, so in that window nothing reaches the stack and
 * little is filed at all.
 *
 * <p>Read through {@link PatchHooks#PRIMARY_ABILITY}, the branch's accessor,
 * as stock Forge publishes none. On a checkout without it, no action reads as in progress, what a special
 * action does is filed as the state-based check would be, and stderr says so
 * once.
 */
final class StackActions {

    static {
        if (!PatchHooks.PRIMARY_ABILITY.present()) {
            System.err.println("effect records: MagicStack.getEffectRecordPrimaryAbility is absent;"
                    + " a special action at priority will be filed as the"
                    + " state-based check after the last cast");
        }
    }

    private StackActions() {
    }

    /**
     * The ability of the item on top of {@code stack}, or null on an empty
     * stack — read through {@code peek()}, because {@code peekAbility()}
     * throws on an empty one.
     */
    static SpellAbility topAbility(MagicStack stack) {
        var top = stack.peek();
        return top == null ? null : top.getSpellAbility();
    }

    /**
     * The ability whose action is in progress on {@code stack}, or null; one
     * marked skipped, a failed payment's, is none (see the class comment for
     * how long that mark lasts).
     */
    static SpellAbility inProgress(MagicStack stack) {
        return PatchHooks.PRIMARY_ABILITY.read(stack) instanceof SpellAbility primary
                && !primary.isSkip() ? primary : null;
    }
}
