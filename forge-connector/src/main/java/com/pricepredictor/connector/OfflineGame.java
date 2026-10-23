package com.pricepredictor.connector;

import forge.card.CardRarity;
import forge.card.CardRules;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.item.PaperCard;

import java.util.List;

/**
 * A player-less game to build cards against outside any match.
 *
 * <p>{@code CardFactory} needs a game to set a card's abilities up against, and
 * with none at all a few cards' setup throws. Converting a script and asking
 * whether a card can be cast both build cards with no match running, so both
 * use this one. Created once, on first use.
 */
final class OfflineGame {

    /**
     * The edition code an offline card's printing carries: none in
     * particular. What is built from it depends on the card's rules, which
     * every printing shares, so the printing is made without looking one up.
     */
    private static final String NO_EDITION = "UNK";

    private OfflineGame() {
    }

    static Game instance() {
        return Holder.INSTANCE;
    }

    /**
     * A card built from {@code rules} against this game, with no owner.
     *
     * <p>The card is built with an explicit, non-negative id. {@code CardFactory}
     * treats a negative id as "for display only" and then reads none of the
     * script's traits — no keywords, no triggers, no statics and no {@code A:}
     * lines, which is where an instant's or sorcery's spell lives — so all
     * that is left are the spells {@code CardState} synthesises from the type
     * line: a permanent's {@code SpellPermanent}, a land's {@code LandAbility}.
     * The three-argument {@code getCard(paper, owner, game)} derives the id
     * from the owner and passes -1 for a null one, so an ownerless card has to
     * be given its id here. Ids come from the game's own counter, which is not
     * thread-safe: offline cards are built on one thread.
     */
    static Card card(CardRules rules) {
        Game game = instance();
        PaperCard paper = new PaperCard(rules, NO_EDITION, CardRarity.Common);
        return CardFactory.getCard(paper, null, game.nextCardId(), game);
    }

    private static final class Holder {
        static final Game INSTANCE = create();

        private static Game create() {
            GameRules rules = new GameRules(GameType.Constructed);
            return new Game(List.of(), rules, new Match(rules, List.of(), "offline"));
        }
    }
}
