package com.pricepredictor.connector;

import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.RegisteredPlayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Plays a sealed match between two decks using Forge's AI and reports per-game detail.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class GamePlayer {

    private static final String LOBBY_NAME_A = "p1";
    private static final String LOBBY_NAME_B = "p2";

    private final int gamesPerMatch;

    /** Per-game outcome: which side won and which side was on the play. */
    public record GameOutcome(String winner, String playFirst) {
        public GameOutcome {
            if (!"A".equals(winner) && !"B".equals(winner)) {
                throw new IllegalArgumentException("winner must be 'A' or 'B', got " + winner);
            }
            if (!"A".equals(playFirst) && !"B".equals(playFirst)) {
                throw new IllegalArgumentException("playFirst must be 'A' or 'B', got " + playFirst);
            }
        }
    }

    /** Full match outcome returned by {@link #playMatch}. */
    public record PlayedMatch(List<GameOutcome> games, int durationSeconds) {}

    /** Create a player with the default best-of-3 match format. */
    public GamePlayer() {
        this(3);
    }

    /** Create a player with a configurable best-of-K match format. */
    public GamePlayer(int gamesPerMatch) {
        this.gamesPerMatch = gamesPerMatch;
    }

    /**
     * Play a match and return per-game detail plus wall-clock duration in seconds.
     * Games in which {@code game.getOutcome().getWinningLobbyPlayer()} is null (draws
     * or aborted games) are skipped and do not appear in the returned list.
     */
    public PlayedMatch playMatch(Deck deckA, Deck deckB) {
        var players = List.of(
                new RegisteredPlayer(deckA).setPlayer(new LobbyPlayerAi(LOBBY_NAME_A, null)),
                new RegisteredPlayer(deckB).setPlayer(new LobbyPlayerAi(LOBBY_NAME_B, null))
        );

        var rules = new GameRules(GameType.Sealed);
        rules.setGamesPerMatch(gamesPerMatch);

        var match = new Match(rules, players, "sealed-match");

        List<GameOutcome> outcomes = new ArrayList<>();
        long startMillis = System.currentTimeMillis();

        while (!match.isMatchOver()) {
            var game = match.createGame();
            match.startGame(game); // blocks until game is finished

            var winningLobby = game.getOutcome().getWinningLobbyPlayer();
            if (winningLobby == null) {
                continue;
            }
            String winner = LOBBY_NAME_A.equals(winningLobby.getName()) ? "A" : "B";

            var startingPlayer = game.getStartingPlayer();
            String playFirst = (startingPlayer != null
                    && LOBBY_NAME_A.equals(startingPlayer.getLobbyPlayer().getName()))
                    ? "A" : "B";

            outcomes.add(new GameOutcome(winner, playFirst));
        }

        int durationSeconds = (int) ((System.currentTimeMillis() - startMillis) / 1000);
        return new PlayedMatch(outcomes, durationSeconds);
    }
}
