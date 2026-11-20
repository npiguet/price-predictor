package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.card.ColorSet;
import forge.game.Game;
import forge.game.card.Card;

import java.util.Objects;

/**
 * {@code colour_type}: set a permanent's colours.
 *
 * <p>Through layer 5 with a fresh timestamp, the way a "becomes blue" effect
 * does, and with no static ability, so the static check does not take it
 * away.
 *
 * @param permanent the fork's permanent
 * @param colours   the colours it becomes, replacing its own
 */
public record ColourTypeOperation(Card permanent, ColorSet colours) implements BoardEdit {

    public ColourTypeOperation {
        Objects.requireNonNull(colours, "colours");
    }

    /** The permanent becomes exactly these colours ({@link ColorSet#C} for colourless). */
    public static ColourTypeOperation colours(Card permanent, ColorSet colours) {
        return new ColourTypeOperation(permanent, colours);
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.COLOUR_TYPE;
    }

    @Override
    public void apply(Game fork) {
        BoardEdit.onBattlefield(permanent);
        permanent.addColor(colours, false, fork.getNextTimestamp(), null);
        BoardEdit.settle(fork);
    }
}
