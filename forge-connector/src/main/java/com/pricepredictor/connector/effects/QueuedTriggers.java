package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;
import forge.game.zone.MagicStack;

import java.util.List;

/**
 * The triggered abilities waiting to be put on a stack (Forge's
 * simultaneous stack list) as they stood at one moment, so that what was
 * queued since can be taken off again, and nothing that was already there.
 *
 * <p>Read through stock Forge only: {@code hasSimultaneousStackEntries},
 * {@code clearSimultaneousStack} and {@code addSimultaneousStackEntry}, and
 * the list itself through {@link ForgeFields#simultaneousStackEntries} where
 * it is not empty. At a priority stop it is empty: the state-based check before
 * priority puts every queued trigger on the stack, so the usual case costs
 * two emptiness checks and no copy.
 */
final class QueuedTriggers {

    private final MagicStack stack;
    /** What waited, compared by identity; null when the list could not be read. */
    private final List<SpellAbility> waiting;

    private QueuedTriggers(MagicStack stack, List<SpellAbility> waiting) {
        this.stack = stack;
        this.waiting = waiting;
    }

    /** What waits on {@code stack} now. */
    static QueuedTriggers of(MagicStack stack) {
        return new QueuedTriggers(stack, stack.hasSimultaneousStackEntries()
                ? ForgeFields.simultaneousStackEntries(stack) : List.of());
    }

    /**
     * Take off every triggered ability queued since, leaving the ones that
     * waited then in their order; nothing when the list could not be read.
     */
    void dropQueuedSince() {
        if (!stack.hasSimultaneousStackEntries() || waiting == null) {
            return;
        }
        if (waiting.isEmpty()) {
            stack.clearSimultaneousStack();
            return;
        }
        List<SpellAbility> now = ForgeFields.simultaneousStackEntries(stack);
        if (now == null) {
            return;
        }
        stack.clearSimultaneousStack();
        now.stream().filter(this::waited).forEach(stack::addSimultaneousStackEntry);
    }

    private boolean waited(SpellAbility queued) {
        return waiting.stream().anyMatch(entry -> entry == queued);
    }
}
