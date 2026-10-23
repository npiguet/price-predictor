package com.pricepredictor.connector.effects;

import forge.StaticData;
import forge.game.Game;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.item.PaperToken;

/**
 * A fresh two-player game for the characterization tests, and the cards on it.
 *
 * <p>A new game per test, never the shared one {@link TestCards} builds in: a
 * card's id is its game's counter, so a card built in a shared game carries an
 * id that depends on how many cards every earlier test built, and the rendered
 * bytes would depend on test order.
 */
final class CharacterizationBoard {

    final Game game;
    /** Player 0, whose turn it is. */
    final Player you;
    final Player opponent;

    private CharacterizationBoard(Game game) {
        this.game = game;
        this.you = game.getPlayers().get(0);
        this.opponent = game.getPlayers().get(1);
    }

    /** An empty board in {@code phase} of the first player's turn, past the start. */
    static CharacterizationBoard empty(PhaseType phase) {
        return new CharacterizationBoard(TestBoard.startedTwoPlayerGame(phase));
    }

    /**
     * A board with something in every inclusion tier the snapshot renders.
     *
     * <p>Yours: an anthem and Cryptolith Rite (statics changing P/T and
     * granting a mana ability), Grizzly Bears with a +1/+1 counter and an
     * attached Bonesplitter, tapped Llanowar Elves, a Forest, Elspeth with the
     * Soldier her +1 made and the emblem her ultimate made, Lightning Bolt in
     * hand, Shock in the graveyard and a Swords to Plowshares in exile. Theirs:
     * a damaged Serra Angel, a Hill Giant Elspeth's other +1 pumped and gave
     * flying until end of turn, a tapped Mountain, Counterspell in hand and
     * Giant Growth in the graveyard. A second Lightning Bolt of yours is on the
     * stack, aimed at the angel.
     */
    static CharacterizationBoard full(PhaseType phase) {
        CharacterizationBoard board = empty(phase);
        Player you = board.you;
        Player opponent = board.opponent;

        board.battlefield("Glorious Anthem", you);
        board.battlefield("Cryptolith Rite", you);
        board.battlefield("Forest", you);
        Card bears = board.battlefield("Grizzly Bears", you);
        bears.addCounterInternal(
                CounterEnumType.P1P1, 1, you, false, null, null);
        board.battlefield("Bonesplitter", you).attachToEntity(bears, null);
        Card elves = board.battlefield("Llanowar Elves", you);
        elves.setTapped(true);
        Card elspeth = board.battlefield("Elspeth, Knight-Errant", you);

        Card angel = board.battlefield("Serra Angel", opponent);
        angel.setDamage(2);
        Card giant = board.battlefield("Hill Giant", opponent);
        board.battlefield("Mountain", opponent).setTapped(true);

        board.resolve(board.ability(elspeth, ApiType.Token), you);
        SpellAbility pump = board.ability(elspeth, ApiType.Pump);
        pump.resetTargets();
        pump.getTargets().add(giant);
        board.resolve(pump, you);
        board.resolve(board.ability(elspeth, ApiType.Effect), you);

        board.hand("Lightning Bolt", you);
        board.graveyard("Shock", you);
        board.game.getAction().exile(board.hand("Swords to Plowshares", you), null, null);
        board.hand("Counterspell", opponent);
        board.graveyard("Giant Growth", opponent);

        board.cast(board.card("Lightning Bolt", you), you, angel);

        board.game.getAction().checkStaticAbilities();
        return board;
    }

    /** The card of this name on the board, wherever it is. */
    Card named(String name) {
        for (Card card : game.getCardsInGame()) {
            if (card.getName().equals(name)) {
                return card;
            }
        }
        throw new AssertionError("no " + name + " in " + game.getCardsInGame());
    }

    Card card(String name, Player owner) {
        return TestBoard.card(game, owner, name);
    }

    /** A token by its script stem, on the battlefield. */
    Card token(String scriptStem, Player owner) {
        PaperToken paper = StaticData.instance().getAllTokens().getToken(scriptStem);
        if (paper == null) {
            throw new AssertionError("token script not in the database: " + scriptStem);
        }
        return game.getAction().moveToPlay(
                CardFactory.getCard(paper, owner, game), owner, null, null);
    }

    Card battlefield(String name, Player owner) {
        return TestBoard.onBattlefield(game, owner, name);
    }

    Card hand(String name, Player owner) {
        return TestBoard.inHand(game, owner, name);
    }

    Card graveyard(String name, Player owner) {
        return TestBoard.inGraveyard(game, owner, name);
    }

    /** A card's spell, activated by {@code activator}. */
    SpellAbility spell(Card card, Player activator) {
        SpellAbility spell = card.getFirstSpellAbility();
        spell.setActivatingPlayer(activator);
        return spell;
    }

    /**
     * Cast a card's spell at a target: onto the stack zone first, then onto
     * the stack, the order the engine's own play path takes.
     */
    SpellAbility cast(Card card, Player caster, Card target) {
        Card onStack = game.getAction().moveToStack(card, null);
        SpellAbility spell = spell(onStack, caster);
        spell.resetTargets();
        spell.getTargets().add(target);
        game.getStack().add(spell);
        return spell;
    }

    /** The card's first ability whose API is {@code api}. */
    SpellAbility ability(Card card, ApiType api) {
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getApi() == api) {
                return sa;
            }
        }
        throw new AssertionError("no " + api + " ability on " + card);
    }

    /** Resolve an ability without the stack: the board it leaves is the point. */
    void resolve(SpellAbility ability, Player activator) {
        ability.setActivatingPlayer(activator);
        AbilityUtils.resolve(ability);
    }
}
