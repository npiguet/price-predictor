package com.pricepredictor.connector.effects.directed.operations;

import forge.StaticData;
import forge.card.CardRules;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.player.Player;
import forge.item.IPaperCard;

/**
 * New cards from the card database, for the edits that put a card where none
 * was: {@link AddPermanentOperation}, {@link ReplaceCardOperation} and
 * {@link MoveCardOperation}'s replacing form.
 */
final class NewCards {

    private NewCards() {
    }

    /**
     * A new card named {@code name}, owned by {@code owner}, in no zone yet.
     * Built as {@code GameAction} builds any card, so it carries its printed
     * traits; nothing has moved it anywhere, so nothing has fired.
     */
    static Card create(Game fork, String name, Player owner) {
        return CardFactory.getCard(paper(name), owner, fork);
    }

    /** The printed rules of the card named {@code name}, read without building a card in any game. */
    static CardRules rules(String name) {
        return paper(name).getRules();
    }

    private static IPaperCard paper(String name) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) {
            throw new IllegalArgumentException("not in the card database: " + name);
        }
        return paper;
    }
}
