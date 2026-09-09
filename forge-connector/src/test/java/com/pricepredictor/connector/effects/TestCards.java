package com.pricepredictor.connector.effects;

import forge.StaticData;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.item.IPaperCard;
import forge.item.PaperToken;

import java.util.List;

/**
 * Real Forge cards for the provenance tests.
 *
 * <p>Built through {@code CardFactory} rather than hand-assembled, because what
 * these tests are about is the shape of the objects the engine actually hands
 * the collectors — a hand-built {@code Card} would have exactly the tidy trait
 * lists whose absence is the defect under test.
 *
 * <p>Requires {@code ForgeExtension} to have initialised the card database.
 */
final class TestCards {

    private TestCards() {
    }

    /**
     * A game exists only because {@code CardFactory} will not build a card
     * without one. No turn is ever taken in it.
     */
    private static final Game GAME = newGame();

    private static int nextCardId = 1;

    private static Game newGame() {
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(List.of(), rules, new Match(rules, List.of(), "ProvenanceTest"));
    }

    static Game game() {
        return GAME;
    }

    /** A fresh card id, for a test that builds a {@code Card} by hand. */
    static int nextCardId() {
        return nextCardId++;
    }

    static Card build(String name) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) {
            throw new AssertionError("card not in the database: " + name);
        }
        return CardFactory.getCard(paper, null, nextCardId++, GAME);
    }

    /**
     * A token, named by its script stem rather than by its printed name: a
     * token is filed under what it is, so the two 1/1 Eldrazi Scions that
     * differ only in a sacrifice ability do not collide.
     */
    static Card token(String scriptStem) {
        PaperToken paper = StaticData.instance().getAllTokens().getToken(scriptStem);
        if (paper == null) {
            throw new AssertionError("token script not in the database: " + scriptStem);
        }
        return CardFactory.getCard(paper, null, nextCardId++, GAME);
    }
}
