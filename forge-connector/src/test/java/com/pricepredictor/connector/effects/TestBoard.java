package com.pricepredictor.connector.effects;

import forge.LobbyPlayer;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameStage;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.item.IPaperCard;

import java.util.ArrayList;
import java.util.List;

/**
 * Fresh games, and real cards placed on them, for the tests that need a board
 * rather than the one shared game {@link TestCards} builds in.
 *
 * <p>Each game is new, so a card's id is its game's own counter and depends on
 * nothing an earlier test built; a test asserting ids keeps them by building
 * its cards in the same order.
 *
 * <p>Requires {@code ForgeExtension} to have initialised the card database.
 */
public final class TestBoard {

    private TestBoard() {
    }

    // ── games ───────────────────────────────────────────────────────────

    /**
     * Two AI seats with empty decks, in {@code phase} of the first seat's
     * turn, not started.
     *
     * <p>Not started means the engine still treats it as being set up: {@code
     * checkStateEffects} returns early and leaves the stack frozen. Use
     * {@link #startedTwoPlayerGame} where a test needs either.
     */
    public static Game twoPlayerGame(PhaseType phase) {
        return twoPlayerGame(phase, new LobbyPlayerAi("a", null), new LobbyPlayerAi("b", null));
    }

    /** The same, past the start: state-based actions run and the stack resolves. */
    public static Game startedTwoPlayerGame(PhaseType phase) {
        Game game = twoPlayerGame(phase);
        game.setAge(GameStage.Play);
        return game;
    }

    /** Two seats of the caller's choosing, in {@code phase} of the first seat's turn. */
    public static Game twoPlayerGame(PhaseType phase, LobbyPlayer first, LobbyPlayer second) {
        Game game = registered(first, second);
        game.getPhaseHandler().devModeSet(phase, game.getPlayers().get(0));
        return game;
    }

    /**
     * Two AI seats under these lobby names and no phase at all: the shape
     * {@code GamePlayer} registers for a match, minus the decks.
     */
    public static Game twoPlayerGame(String first, String second) {
        return registered(new LobbyPlayerAi(first, null), new LobbyPlayerAi(second, null));
    }

    private static Game registered(LobbyPlayer first, LobbyPlayer second) {
        Deck deck = new Deck();
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(deck).setPlayer(first),
                new RegisteredPlayer(deck).setPlayer(second));
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(players, rules, new Match(rules, players, "TestBoard"));
    }

    /**
     * A game with no players that is not {@link TestCards#game()}: what a fork
     * is to the live game a collector was built over ({@code GameCopier}
     * builds every copied card against a new {@code Game}), or a game whose
     * per-game state a test changes without leaking it into the shared one.
     */
    static Game anotherGame() {
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(List.of(), rules, new Match(rules, List.of(), "another game"));
    }

    // ── cards ───────────────────────────────────────────────────────────

    /** A real card owned by {@code owner}, in no zone yet. */
    public static Card card(Game game, Player owner, String name) {
        return CardFactory.getCard(paper(name), owner, game);
    }

    /** A card that has just entered the battlefield, summoning sick. */
    public static Card onBattlefield(Game game, Player owner, String name) {
        return game.getAction().moveToPlay(card(game, owner, name), owner, null, null);
    }

    /**
     * A card on the battlefield since the turn began, so a creature may attack
     * and tap.
     */
    public static Card inPlay(Game game, Player owner, String name) {
        Card card = card(game, owner, name);
        game.getAction().moveToPlay(card, owner, null, null);
        card.setSickness(false);
        return card;
    }

    public static Card inHand(Game game, Player owner, String name) {
        return game.getAction().moveTo(ZoneType.Hand, card(game, owner, name), null, null);
    }

    static Card inGraveyard(Game game, Player owner, String name) {
        return game.getAction().moveTo(ZoneType.Graveyard, card(game, owner, name), null, null);
    }

    /**
     * An unowned card in {@code game}, numbered from {@link TestCards}' counter
     * rather than the game's, as a card built for the shared game is.
     */
    static Card cardIn(Game game, String name) {
        return CardFactory.getCard(paper(name), null, TestCards.nextCardId(), game);
    }

    /** An unowned card in a game of its own; see {@link #anotherGame()}. */
    static Card cardInAnotherGame(String name) {
        return cardIn(anotherGame(), name);
    }

    private static IPaperCard paper(String name) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) {
            throw new AssertionError("card not in the database: " + name);
        }
        return paper;
    }

    // ── charms ──────────────────────────────────────────────────────────

    /** The card's charm spell, activated by {@code activator}. */
    static SpellAbility charmOf(Card card, Player activator) {
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getApi() == ApiType.Charm) {
                sa.setActivatingPlayer(activator);
                return sa;
            }
        }
        throw new AssertionError("no charm on " + card);
    }

    /**
     * Chain the modes at these {@code Choices$} positions, as the cast would,
     * and return the clones that now hang off the charm, in resolution order.
     */
    static List<AbilitySub> chain(SpellAbility charm, int... positions) {
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        List<AbilitySub> chosen = new ArrayList<>();
        for (int position : positions) {
            chosen.add(choices.get(position));
        }
        charm.setChosenList(chosen);
        CharmEffect.chainAbilities(charm, chosen);
        List<AbilitySub> clones = new ArrayList<>();
        for (SpellAbility sub = charm.getSubAbility(); sub != null; sub = sub.getSubAbility()) {
            clones.add((AbilitySub) sub);
        }
        return clones;
    }
}
