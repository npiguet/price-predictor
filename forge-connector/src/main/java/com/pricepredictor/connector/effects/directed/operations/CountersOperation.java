package com.pricepredictor.connector.effects.directed.operations;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.player.Player;

/**
 * {@code counters}: set how many counters of one kind a permanent or a player
 * has.
 *
 * <p>The counter table is written directly: putting counters on or removing
 * them in play fires counter-added and counter-removed triggers and
 * replacements (Hardened Scales, Doubling Season) and adds to the counters
 * put this turn, none of which a variant states. A card's whole table is
 * handed back at once, so a keyword counter (a flying counter) still grants
 * its keyword.
 *
 * @param entity the fork's permanent or player
 * @param type   the counter kind
 * @param count  how many it has afterwards, 0 for none
 */
public record CountersOperation(GameEntity entity, CounterType type, int count) implements BoardEdit {

    public CountersOperation {
        if (count < 0) {
            throw new IllegalArgumentException("a counter count is not negative: " + count);
        }
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.COUNTERS;
    }

    @Override
    public void apply(Game fork) {
        if (entity instanceof Card card) {
            BoardEdit.onBattlefield(card);
            Multiset<CounterType> counters = HashMultiset.create(card.getCounters());
            counters.setCount(type, count);
            card.setCounters(counters);
        } else if (entity instanceof Player player) {
            player.setCounters(type, count, null, false);
        } else {
            throw new IllegalStateException("counters on neither a card nor a player: " + entity);
        }
        BoardEdit.settle(fork);
    }
}
