package com.pricepredictor.connector;

import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.RegisteredPlayer;

import java.util.List;

/**
 * Plays a sealed match between two decks using Forge's AI.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class GamePlayer {

    private final int gamesPerMatch;

    /** Create a player with the default best-of-3 match format. */
    public GamePlayer() {
        this(3);
    }

    /** Create a player with a configurable best-of-K match format. */
    public GamePlayer(int gamesPerMatch) {
        this.gamesPerMatch = gamesPerMatch;
    }

    /**
     * Play a match and return the win counts.
     *
     * @param deckA first deck
     * @param deckB second deck
     * @return int[]{winsA, winsB}
     */
    public int[] playMatch(Deck deckA, Deck deckB) {
        var players = List.of(
                new RegisteredPlayer(deckA).setPlayer(new LobbyPlayerAi("p1", null)),
                new RegisteredPlayer(deckB).setPlayer(new LobbyPlayerAi("p2", null))
        );

        var rules = new GameRules(GameType.Sealed);
        rules.setGamesPerMatch(gamesPerMatch);

        var match = new Match(rules, players, "sealed-match");

        int winsA = 0;
        int winsB = 0;

        while (!match.isMatchOver()) {
            var game = match.createGame();
            match.startGame(game); // blocks until game is finished

            var winner = game.getOutcome().getWinningLobbyPlayer();
            if (winner != null) {
                if (winner.getName().equals("p1")) {
                    winsA++;
                } else {
                    winsB++;
                }
            }
        }

        return new int[]{winsA, winsB};
    }
}
