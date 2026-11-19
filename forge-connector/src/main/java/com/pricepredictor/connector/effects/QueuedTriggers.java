package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;
import forge.game.zone.MagicStack;

import java.util.List;

/**
 * The triggered abilities waiting to be put on a stack (Forge's
 * simultaneous stack list) as they stood at one moment, so that what was
 * queued since can be taken off again, and nothing that was already there.
 *
 * <p>The list is changed through stock Forge's {@code clearSimultaneousStack}
 * and {@code addSimultaneousStackEntry}, and read through the branch's
 * {@link PatchHooks#SIMULTANEOUS_STACK_ENTRIES}, a copy, only where it was
 * not empty. At a priority stop it is empty: the state-based check before
 * priority puts every queued trigger on the stack, so the usual case costs
 * two emptiness checks and no read. A checkout without the accessor, which
 * {@link PatchHooks#report()} names, reads every list as empty, so a list
 * that was not empty is left as it is.
 */
final class QueuedTriggers {

    private final MagicStack stack;
    private final boolean wasEmpty;
    /** What waited, compared by identity. */
    private final List<SpellAbility> waiting;

    private QueuedTriggers(MagicStack stack, boolean wasEmpty, List<SpellAbility> waiting) {
        this.stack = stack;
        this.wasEmpty = wasEmpty;
        this.waiting = waiting;
    }

    /** What waits on {@code stack} now. */
    static QueuedTriggers of(MagicStack stack) {
        boolean empty = !stack.hasSimultaneousStackEntries();
        return new QueuedTriggers(stack, empty, empty ? List.of() : entries(stack));
    }

    /**
     * Take off every triggered ability queued since, leaving the ones that
     * waited then in their order.
     */
    void dropQueuedSince() {
        if (!stack.hasSimultaneousStackEntries()) {
            return;
        }
        if (wasEmpty) {
            stack.clearSimultaneousStack();
            return;
        }
        List<SpellAbility> now = entries(stack);
        List<SpellAbility> kept = now.stream().filter(this::waited).toList();
        if (kept.size() == now.size()) {
            return;
        }
        stack.clearSimultaneousStack();
        kept.forEach(stack::addSimultaneousStackEntry);
    }

    private boolean waited(SpellAbility queued) {
        return waiting.stream().anyMatch(entry -> entry == queued);
    }

    /** The branch's copy of the list, in order; empty without the accessor. */
    private static List<SpellAbility> entries(MagicStack stack) {
        return PatchHooks.SIMULTANEOUS_STACK_ENTRIES.read(stack) instanceof List<?> queued
                ? queued.stream().map(SpellAbility.class::cast).toList() : List.of();
    }
}
