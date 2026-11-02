package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;

/**
 * {@code pt}: set a creature's base power, toughness or both.
 *
 * <p>A setting effect in layer 7b with a fresh timestamp, the way a "becomes
 * a 3/3" effect sets it: it applies after every earlier setting effect, and
 * the boosts and counters of layers 7c and 7d still apply on top, so a
 * variant's "toughness 3 instead of 2" changes the base and nothing else. It
 * carries no static ability, so the static check does not take it away.
 *
 * @param creature  the fork's creature
 * @param power     the base power to set, or null to keep the creature's
 * @param toughness the base toughness to set, or null to keep the creature's
 */
public record PtOperation(Card creature, Integer power, Integer toughness) implements BoardEdit {

    public PtOperation {
        if (power == null && toughness == null) {
            throw new IllegalArgumentException("a pt edit sets power, toughness or both");
        }
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.PT;
    }

    @Override
    public void apply(Game fork) {
        if (!BoardEdit.onBattlefield(creature).isCreature()) {
            throw new IllegalStateException("not a creature: " + creature);
        }
        creature.addNewPT(power, toughness, fork.getNextTimestamp(), 0);
        BoardEdit.settle(fork);
    }
}
