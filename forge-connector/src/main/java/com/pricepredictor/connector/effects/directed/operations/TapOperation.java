package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;

/**
 * {@code tap}: tap or untap a permanent.
 *
 * <p>The tapped flag is set directly: tapping or untapping in play fires the
 * becomes-tapped and becomes-untapped triggers and replacements, which a
 * variant's "the creature is tapped" does not state.
 *
 * @param permanent the fork's permanent
 * @param tapped    whether it is tapped afterwards
 */
public record TapOperation(Card permanent, boolean tapped) implements BoardEdit {

    @Override
    public ForkOperation operation() {
        return ForkOperation.TAP;
    }

    @Override
    public void apply(Game fork) {
        BoardEdit.onBattlefield(permanent).setTapped(tapped);
        BoardEdit.settle(fork);
    }
}
