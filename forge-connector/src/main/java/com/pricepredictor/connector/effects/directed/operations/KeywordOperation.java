package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
import forge.game.card.Card;

import java.util.List;

/**
 * {@code keyword}: give a permanent a keyword or take one away, through
 * {@link KeywordEdit}, the edit the damage-step probe strips a combatant's
 * keyword with.
 *
 * @param permanent the fork's permanent
 * @param keyword   to add, spelled as a card script spells it ({@code Flying},
 *                  {@code Ward:2}); to remove, in that spelling or the
 *                  corpus's ({@code first_strike}), every instance the card
 *                  carries going
 * @param add       whether to add the keyword rather than remove it
 */
public record KeywordOperation(Card permanent, String keyword, boolean add) implements BoardEdit {

    public static KeywordOperation adding(Card permanent, String keyword) {
        return new KeywordOperation(permanent, keyword, true);
    }

    public static KeywordOperation removing(Card permanent, String keyword) {
        return new KeywordOperation(permanent, keyword, false);
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.KEYWORD;
    }

    @Override
    public void apply(Game fork) {
        BoardEdit.onBattlefield(permanent);
        if (add) {
            KeywordEdit.add(fork, permanent, keyword);
        } else {
            List<String> carried = KeywordEdit.carried(permanent, keyword);
            if (carried.isEmpty()) {
                throw new IllegalStateException(permanent + " carries no " + keyword);
            }
            for (String original : carried) {
                KeywordEdit.remove(fork, permanent, original);
            }
        }
        BoardEdit.settle(fork);
    }
}
