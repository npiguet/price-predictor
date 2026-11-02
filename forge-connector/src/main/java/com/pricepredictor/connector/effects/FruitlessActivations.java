package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The free activations whose resolution changed nothing this turn, which the
 * random seat does not draw again until the turn ends (FR-040, research C9).
 *
 * <p>The seat never passes while something is playable, so a free repeatable
 * ability that does nothing — a Gigantoplasm setting its base to 0/0 under a
 * +2/+2 boost — would be activated until the game's own limits stop it, and
 * fill a corpus with one record. The collector is what sees that a
 * resolution did nothing, so it reports it here, and the seat filters what
 * it reports from its candidates.
 *
 * <p>A free activation still on the stack counts too. The activating player
 * keeps priority, so the seat would stack the same free activation again and
 * again before the first resolved and showed whether it changes anything; it
 * waits for that resolution instead.
 *
 * <p>Keyed by the card and the activated ability's printed line, as the
 * seat's candidates name them; an activation whose line cannot be keyed is not
 * remembered, since nothing could match it. The turn's set is cleared at the
 * first report or query of a new turn.
 */
public final class FruitlessActivations {

    private record Activation(int cardId, ProvenanceKey key) {
    }

    private final Game game;
    private final Set<Activation> thisTurn = new HashSet<>();
    /** The free activations on the stack, by the id of the ability there. */
    private final Map<Integer, Activation> onStack = new HashMap<>();
    private int turn = Integer.MIN_VALUE;

    FruitlessActivations(Game game) {
        this.game = game;
    }

    /**
     * Whether a resolution's effect halves changed nothing: no event at all,
     * or only an {@code ability_change} that granted and removed nothing — an
     * {@code Animate} that sets power and toughness alone writes one such,
     * whether or not the stats moved, and a stat that moved is an event of
     * its own.
     */
    static boolean changesNothing(List<EffectEvent> events) {
        for (EffectEvent event : events) {
            if (!(event.type() == EventType.ABILITY_CHANGE
                    && "".equals(String.valueOf(event.params().getOrDefault("abilities", "")))
                    && !Boolean.TRUE.equals(event.params().get("removed")))) {
                return false;
            }
        }
        return true;
    }

    /** A free activation of the line {@code key} has been put on the stack. */
    void pending(SpellAbility onStackAbility, ProvenanceKey key) {
        Card host = onStackAbility.getHostCard();
        if (host == null || key == null) {
            return;
        }
        onStack.put(onStackAbility.getId(), new Activation(host.getId(), rootOf(key)));
    }

    /** {@code host}'s activation of the line {@code key} changed nothing. */
    void report(Card host, ProvenanceKey key) {
        if (host == null || key == null) {
            return;
        }
        current();
        thisTurn.add(new Activation(host.getId(), rootOf(key)));
    }

    /**
     * Whether {@code ability}, a candidate of the seat's priority stop, is an
     * activation that changed nothing earlier this turn, or a free one whose
     * earlier activation is still on the stack.
     */
    public boolean isFruitless(SpellAbility ability) {
        current();
        dropResolved();
        if ((thisTurn.isEmpty() && onStack.isEmpty())
                || ability == null || ability.getHostCard() == null) {
            return false;
        }
        ProvenanceKey key = ProvenanceResolver.of(ability);
        if (key == null) {
            return false;
        }
        Activation activation = new Activation(ability.getHostCard().getId(), rootOf(key));
        return thisTurn.contains(activation) || onStack.containsValue(activation);
    }

    /** Forget the pending activations that have left the stack, however they left. */
    private void dropResolved() {
        if (onStack.isEmpty()) {
            return;
        }
        Set<Integer> ids = new HashSet<>();
        game.getStack().forEach(instance -> {
            if (instance.getSpellAbility() != null) {
                ids.add(instance.getSpellAbility().getId());
            }
        });
        onStack.keySet().retainAll(ids);
    }

    /** The line itself, whichever of its modes resolved. */
    private static ProvenanceKey rootOf(ProvenanceKey key) {
        return new ProvenanceKey(
                key.scriptFile(), key.face(), key.traitKind(), key.indexWithinKind());
    }

    private void current() {
        int now = game.getPhaseHandler().getTurn();
        if (now != turn) {
            thisTurn.clear();
            turn = now;
        }
    }
}
