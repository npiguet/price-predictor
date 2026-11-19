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
 * What a copy copied, as it was when the copy was made (finding 1.8). A
 * Clone takes the copiable values of what it copies when it enters (707.2):
 * a later copy effect on the original (Cytoshape on Moldering Karok) adds a
 * clone state to the original, not to the Clone, so the original's latest
 * clone state, which {@code Card.getState} reads, is no longer what the Clone
 * copied. Each clone state is filed under its effect's timestamp, so what
 * the original showed when the Clone copied it is its latest clone state
 * filed before the Clone's.
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
        List<Card> chain = new ArrayList<>(List.of(card));
        long before = NOW;
        for (int depth = 0; depth < MAX_COPY_DEPTH; depth++) {
            Map.Entry<Long, CardCloneStates> copy = copyBefore(chain.get(chain.size() - 1), before);
            Card origin = copy == null ? null : copy.getValue().getOrigin();
            if (origin == null) {
                break;
            }
            chain.add(origin);
            before = copy.getKey();
        }
        return chain;
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
