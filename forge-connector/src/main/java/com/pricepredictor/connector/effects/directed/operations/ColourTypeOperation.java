package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.game.Game;
import forge.game.card.Card;

import java.util.EnumSet;
import java.util.List;

/**
 * {@code colour_type}: set a permanent's colours, or give it a type or take
 * one away.
 *
 * <p>Through layers 5 and 4 with a fresh timestamp, the way a "becomes blue"
 * or "becomes an artifact in addition to its other types" effect does, and
 * with no static ability, so the static check does not take it away. One
 * edit changes one fact: either the colours or one type.
 *
 * @param permanent   the fork's permanent
 * @param colours     the colours it becomes, replacing its own, or null
 * @param addedType   a type, subtype or supertype it gains ({@code Artifact},
 *                    {@code Zombie}), or null
 * @param removedType one it loses, or null
 */
public record ColourTypeOperation(
        Card permanent, ColorSet colours, String addedType, String removedType)
        implements BoardEdit {

    public ColourTypeOperation {
        int set = (colours != null ? 1 : 0) + (addedType != null ? 1 : 0)
                + (removedType != null ? 1 : 0);
        if (set != 1) {
            throw new IllegalArgumentException(
                    "a colour_type edit changes the colours or one type, not " + set + " facts");
        }
    }

    /** The permanent becomes exactly these colours ({@link ColorSet#C} for colourless). */
    public static ColourTypeOperation colours(Card permanent, ColorSet colours) {
        return new ColourTypeOperation(permanent, colours, null, null);
    }

    public static ColourTypeOperation addingType(Card permanent, String type) {
        return new ColourTypeOperation(permanent, null, type, null);
    }

    public static ColourTypeOperation removingType(Card permanent, String type) {
        return new ColourTypeOperation(permanent, null, null, type);
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.COLOUR_TYPE;
    }

    @Override
    public void apply(Game fork) {
        BoardEdit.onBattlefield(permanent);
        long timestamp = fork.getNextTimestamp();
        if (colours != null) {
            permanent.addColor(colours, false, timestamp, null);
        } else {
            permanent.addChangedCardTypes(
                    addedType == null ? null : new CardType(List.of(addedType), true),
                    removedType == null ? null : new CardType(List.of(removedType), true),
                    false, EnumSet.noneOf(RemoveType.class), timestamp, 0, true, false);
        }
        BoardEdit.settle(fork);
    }
}
