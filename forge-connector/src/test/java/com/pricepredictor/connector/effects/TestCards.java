package com.pricepredictor.connector.effects;

import forge.StaticData;
import forge.card.GamePieceType;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.item.IPaperCard;
import forge.item.PaperToken;


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
    private static final Game GAME = TestBoard.anotherGame();

    private static int nextCardId = 1;

    static Game game() {
        return GAME;
    }

    /** A fresh card id, for a test that builds a {@code Card} by hand. */
    static int nextCardId() {
        return nextCardId++;
    }

    /** A card's paper printing, failing the test when the database has none. */
    static IPaperCard paper(String name) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) {
            throw new AssertionError("card not in the database: " + name);
        }
        return paper;
    }

    /** A token's paper printing by its script stem, failing the test when there is none. */
    static PaperToken paperToken(String scriptStem) {
        PaperToken paper = StaticData.instance().getAllTokens().getToken(scriptStem);
        if (paper == null) {
            throw new AssertionError("token script not in the database: " + scriptStem);
        }
        return paper;
    }

    static Card build(String name) {
        return CardFactory.getCard(paper(name), null, nextCardId++, GAME);
    }

    /**
     * A token, named by its script stem rather than by its printed name: a
     * token is filed under what it is, so the two 1/1 Eldrazi Scions that
     * differ only in a sacrifice ability do not collide.
     */
    static Card token(String scriptStem) {
        return CardFactory.getCard(paperToken(scriptStem), null, nextCardId++, GAME);
    }

    /**
     * A token the way {@code GameCopier} rebuilds one in a forked game:
     * {@code TokenInfo.toCard} makes a bare {@code Card}, copies the name,
     * image key, colour, types and P/T, and never sets a paper card. Every
     * ability-bearing token in a forked combat record arrives like this.
     */
    static Card copiedToken(String scriptStem) {
        Card original = token(scriptStem);
        Card copy = new Card(nextCardId++, GAME);
        copy.setName(original.getName());
        copy.setImageKey(original.getImageKey());
        copy.setGamePieceType(GamePieceType.TOKEN);
        for (forge.card.CardType.CoreType type : original.getType().getCoreTypes()) {
            copy.addType(type.toString());
        }
        for (String subtype : original.getType().getSubtypes()) {
            copy.addType(subtype);
        }
        copy.setBasePower(original.getBasePower());
        copy.setBaseToughness(original.getBaseToughness());
        return copy;
    }

    /** The flashback keyword a card state prints, Blast from the Past's for one. */
    static KeywordInterface flashbackOf(CardState state) {
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Flashback")) {
                return keyword;
            }
        }
        throw new AssertionError("no flashback on " + state.getCard().getName());
    }

    /**
     * The first ability on a card whose API matches, for the emitter tests:
     * each top-level ability is searched, its sub-ability chain included,
     * before the next.
     */
    static SpellAbility scriptedAbility(String cardName, String api) {
        Card card = build(cardName);
        for (SpellAbility sa : card.getCurrentState().getSpellAbilities()) {
            if (sa.getApi() != null && api.equals(sa.getApi().name())) {
                return sa;
            }
            for (SpellAbility sub = sa.getSubAbility(); sub != null; sub = sub.getSubAbility()) {
                if (sub.getApi() != null && api.equals(sub.getApi().name())) {
                    return sub;
                }
            }
        }
        throw new AssertionError("no " + api + " ability on " + cardName);
    }
}
