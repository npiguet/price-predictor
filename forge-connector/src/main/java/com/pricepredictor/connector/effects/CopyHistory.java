package com.pricepredictor.connector.effects;

import forge.card.CardStateName;
import forge.game.card.Card;
import forge.game.card.CardCloneStates;
import forge.game.card.CardState;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;

/**
 * What a copy copied, as it was when the copy was made. A
 * Clone takes the copiable values of what it copies when it enters (707.2):
 * a later copy effect on the original (Cytoshape on Moldering Karok) adds a
 * clone state to the original, not to the Clone, so the original's latest
 * clone state, which {@code Card.getState} reads, is no longer what the Clone
 * copied. Each clone state is filed under its effect's timestamp, so what
 * the original showed when the Clone copied it is its latest clone state
 * filed before the Clone's. A copy effect on the original that has since
 * ended (Cytoshape's, until end of turn) took its clone state away, so the
 * walk then misses what the Clone copied through it: the patch notes on the
 * Clone's clone states the card whose printed values they took, read when
 * the Clone copied, and the walk follows that card when it misses it.
 *
 * <p>A copy effect's clone states are those {@code Card.isCloned} counts as
 * a copy: neither a mutation's merged state nor a prototype's.
 */
final class CopyHistory {

    /** A time after every clone state: the card as it is now. */
    static final long NOW = Long.MAX_VALUE;

    /** How many copies of copies {@link #chainOf} follows before giving up. */
    private static final int MAX_COPY_DEPTH = 8;

    private CopyHistory() {
    }

    /**
     * The copy effect's clone states, filed under their timestamp, that
     * hold {@code state}; null for a state no copy effect's clone states
     * hold.
     */
    static Map.Entry<Long, CardCloneStates> copyHolding(CardState state) {
        Card host = state == null ? null : state.getCard();
        if (host == null || !(host.getCloneStates() instanceof NavigableMap<Long, CardCloneStates> states)) {
            return null;
        }
        for (Map.Entry<Long, CardCloneStates> entry : states.descendingMap().entrySet()) {
            if (isCopy(host, entry.getKey()) && holds(entry.getValue(), state)) {
                return entry;
            }
        }
        return null;
    }

    /**
     * The state of {@code name} {@code card} showed just before
     * {@code before}, as {@code Card.getState} read it then: its latest
     * clone states filed earlier, a mutation's or a prototype's included,
     * else its printed state. Null when those clone states hold no state of
     * that name.
     */
    static CardState stateAsOf(Card card, CardStateName name, long before) {
        Map.Entry<Long, CardCloneStates> entry = card.getCloneStates() instanceof NavigableMap<Long,
                CardCloneStates> states ? states.lowerEntry(before) : null;
        return entry == null ? card.getOriginalState(name) : entry.getValue().get(name);
    }

    /**
     * The card whose printed lines {@code card} prints: the last of
     * {@link #chainOf}, {@code card} itself when it is no copy.
     */
    static Card printerOf(Card card) {
        List<Card> chain = chainOf(card);
        return chain.get(chain.size() - 1);
    }

    /**
     * {@code card}, then what it copies: its latest copy effect's origin,
     * and each origin's copy effect filed before the one that copied it,
     * to a card that was no copy then. A mutated permanent is read through
     * the copy effect under its mutation: a Clone that mutated still prints
     * what it copied among the abilities it brought. A copy effect's added
     * keyword (707.9b) is printed by the link whose copy effect added it.
     */
    static List<Card> chainOf(Card card) {
        return chainFrom(card, NOW, 0);
    }

    /**
     * The card a copy effect's clone states copied, as it was then: their
     * origin, else, when the walk from it no longer reaches the card whose
     * printed values they took ({@link #printerNoted}), that card.
     */
    static Card originOf(Map.Entry<Long, CardCloneStates> copy) {
        List<Card> tail = tailOf(copy, 0);
        return tail.isEmpty() ? null : tail.get(0);
    }

    /** {@code card}, then what its latest copy effect filed before {@code before} copied, and so on. */
    private static List<Card> chainFrom(Card card, long before, int depth) {
        List<Card> chain = new ArrayList<>(List.of(card));
        chain.addAll(tailOf(copyBefore(card, before), depth));
        return chain;
    }

    /**
     * What {@code copy} copied, then what that copied when {@code copy} was
     * made, and so on; empty for no copy, or past {@link #MAX_COPY_DEPTH}.
     * The walk reads the origin's clone states, which an ended copy effect
     * on the origin (Cytoshape's, until end of turn) took away, so when it
     * misses the card the clone states noted as their printer when they
     * were built, the walk is that card's instead (707.2).
     */
    private static List<Card> tailOf(Map.Entry<Long, CardCloneStates> copy, int depth) {
        Card origin = copy == null ? null : copy.getValue().getOrigin();
        if (origin == null || depth >= MAX_COPY_DEPTH) {
            return List.of();
        }
        List<Card> tail = chainFrom(origin, copy.getKey(), depth + 1);
        Card printer = printerNoted(copy.getValue());
        return printer == null || tail.contains(printer) ? tail : chainFrom(printer, copy.getKey(), depth + 1);
    }

    /**
     * The card whose printed values {@code states} took, as the patch's
     * {@code CardCloneStates.getEffectRecordPrinter} noted it when they were
     * built; null on a checkout without the hook, and for clone states no
     * copy effect built.
     */
    private static Card printerNoted(CardCloneStates states) {
        return PatchHooks.CLONE_PRINTER.read(states) instanceof Card printer ? printer : null;
    }

    /** {@code card}'s latest copy effect's clone states filed before {@code before}; null for none. */
    private static Map.Entry<Long, CardCloneStates> copyBefore(Card card, long before) {
        if (!(card.getCloneStates() instanceof NavigableMap<Long, CardCloneStates> states)) {
            return null;
        }
        for (Map.Entry<Long, CardCloneStates> entry = states.lowerEntry(before); entry != null;
                entry = states.lowerEntry(entry.getKey())) {
            if (isCopy(card, entry.getKey())) {
                return entry;
            }
        }
        return null;
    }

    /** Whether the clone states {@code card} files under {@code timestamp} are a copy effect's. */
    private static boolean isCopy(Card card, long timestamp) {
        return timestamp != card.getMutatedTimestamp() && timestamp != card.getPrototypeTimestamp();
    }

    private static boolean holds(CardCloneStates states, CardState state) {
        return states.values().stream().anyMatch(each -> each == state);
    }
}
