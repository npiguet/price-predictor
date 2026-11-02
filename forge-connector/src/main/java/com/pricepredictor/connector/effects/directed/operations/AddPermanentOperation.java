package com.pricepredictor.connector.effects.directed.operations;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Map;
import java.util.Objects;

/**
 * {@code add_permanent}: put a new card from the card database onto the
 * battlefield (FR-011a).
 *
 * <p>Built under its owner, given a fresh timestamp as anything entering a
 * zone is (613.7d), added to its controller's battlefield list, then tapped,
 * summoning-sick and countered as the variant states. Nothing moved it there:
 * no enters trigger fires (an Elvish Visionary draws nothing), no
 * enters-the-battlefield replacement applies (it does not enter tapped or with
 * counters unless the variant says so), and no per-turn count moves. Its
 * static abilities apply from the static check this edit ends with.
 *
 * @param cardName   the card's name in the card database
 * @param owner      the fork's player who owns it
 * @param controller the fork's player whose battlefield it is put on
 * @param tapped     whether it is tapped
 * @param sick       whether it is summoning-sick (came under its controller's
 *                   control this turn)
 * @param counters   the counters it has, by kind; empty for none
 */
public record AddPermanentOperation(
        String cardName, Player owner, Player controller, boolean tapped, boolean sick,
        Map<CounterType, Integer> counters) implements BoardEdit {

    public AddPermanentOperation {
        Objects.requireNonNull(cardName, "cardName");
        Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(controller, "controller");
        counters = Map.copyOf(counters);
    }

    /** An untapped card with no counters, owned and controlled by {@code player}. */
    public static AddPermanentOperation of(String cardName, Player player, boolean sick) {
        return new AddPermanentOperation(cardName, player, player, false, sick, Map.of());
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.ADD_PERMANENT;
    }

    @Override
    public void apply(Game fork) {
        Card card = NewCards.create(fork, cardName, owner);
        put(fork, card, controller, null);
        card.setTapped(tapped);
        card.setSickness(sick);
        Multiset<CounterType> table = HashMultiset.create();
        counters.forEach(table::setCount);
        card.setCounters(table);
        BoardEdit.settle(fork);
    }

    /**
     * The adding step, shared with {@link ReplaceCardOperation}: a fresh
     * timestamp, the controller where it is not the owner, and the raw add to
     * the controller's battlefield list.
     *
     * @param index the card's place in that list, or null for its end
     * @return the card, now on the battlefield
     */
    static Card put(Game fork, Card card, Player controller, Integer index) {
        long timestamp = fork.getNextTimestamp();
        card.setGameTimestamp(timestamp);
        if (!controller.equals(card.getOwner())) {
            card.setController(controller, timestamp);
        }
        controller.getZone(ZoneType.Battlefield).add(card, index);
        return card;
    }
}
