package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.BusBracketCollector;
import com.pricepredictor.connector.effects.ForkCollector;
import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.RecordShardWriter;
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

    static final String LOBBY_NAME_A = "p1";
    static final String LOBBY_NAME_B = "p2";

    private final int gamesPerMatch;

    /**
     * Where effect records go, or null when the run is not instrumented.
     *
     * <p>Absent, nothing changes: no collector is subscribed, no shard is
     * opened, and the match writes exactly what it always did. That is what
     * makes the instrumentation an opt-in rather than a mode.
     */
    private final RecordShardWriter effectRecords;

    /**
     * Per-game outcome: which side won, which side was on the play, and the
     * non-basic, non-token card names that each side played during that game
     * (controlled by the side that owns them, paperCard names, multiplicities
     * preserved).
     */
    public record GameOutcome(
            String winner,
            String playFirst,
            List<String> cardsPlayedA,
            List<String> cardsPlayedB) {
        public GameOutcome {
            if (!"A".equals(winner) && !"B".equals(winner)) {
                throw new IllegalArgumentException("winner must be 'A' or 'B', got " + winner);
            }
            if (!"A".equals(playFirst) && !"B".equals(playFirst)) {
                throw new IllegalArgumentException("playFirst must be 'A' or 'B', got " + playFirst);
            }
            cardsPlayedA = List.copyOf(cardsPlayedA);
            cardsPlayedB = List.copyOf(cardsPlayedB);
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
        this(gamesPerMatch, null);
    }

    /**
     * Create a player that also collects effect records into {@code writer}.
     *
     * <p>The instrumentation costs no extra simulation: it rides matches that
     * were going to be played anyway.
     */
    public GamePlayer(int gamesPerMatch, RecordShardWriter effectRecords) {
        this.gamesPerMatch = gamesPerMatch;
        this.effectRecords = effectRecords;
    }

    /**
     * Play a match and return per-game detail plus wall-clock duration in seconds.
     * Games in which {@code game.getOutcome().getWinningLobbyPlayer()} is null (draws
     * or aborted games) are skipped and do not appear in the returned list.
     *
     * <p>A fresh {@link PlayedCardCollector} is registered as an event-bus
     * visitor on every game so the per-game card play is captured for
     * {@code cards-played.txt}.
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
        // Seeds the patched collectors' sampling, so a worker's games sample
        // independently of one another rather than all alike.
        long gameSeed = startMillis;
        // Read once per match rather than per game: the supervisor sets them
        // when it spawns the worker and they do not change under it.
        PatchedCollectors.CollectionCaps caps =
                PatchedCollectors.CollectionCaps.fromSystemProperties();

        while (!match.isMatchOver()) {
            var game = match.createGame();
            var collector = new PlayedCardCollector();
            game.subscribeToEvents(collector);
            PatchedCollectors patched = null;
            if (effectRecords != null) {
                // One collector per game, with the game's own id: records join
                // to a checkpoint's recorded split by game_id, so the id has to
                // change when the game does. Both collectors share that id —
                // they describe the same game.
                String gameId = effectRecords.nextGameId();
                game.subscribeToEvents(new BusBracketCollector(
                        game, effectRecords, gameId));
                // The patched collectors reach the engine through static
                // listeners, so they are installed for the life of one game and
                // uninstalled after it. On a stock checkout install() finds no
                // hook and the game plays exactly as it did before.
                patched = new PatchedCollectors(
                        game, effectRecords, gameId, caps, gameSeed);
                if (caps.interventionsPerGame() > 0 || caps.probesEnabled()) {
                    // Stage three. Nothing is copied unless a budget says so,
                    // because a fork costs a game copy and a stack resolution.
                    patched.withForks(new ForkCollector(
                            game, effectRecords, gameId, caps, gameSeed));
                }
                gameSeed++;
                patched.install();
                // Continuous effects are read off the layer tables at a phase
                // boundary, which is the cheapest moment the board is stable.
                game.subscribeToEvents(patched.phaseBridge());
            }
            try {
                match.startGame(game); // blocks until game is finished
            } finally {
                if (patched != null) {
                    patched.close();
                }
            }

            var winningLobby = game.getOutcome().getWinningLobbyPlayer();
            if (winningLobby == null) {
                continue;
            }
            String winner = LOBBY_NAME_A.equals(winningLobby.getName()) ? "A" : "B";

            var startingPlayer = game.getStartingPlayer();
            String playFirst = (startingPlayer != null
                    && LOBBY_NAME_A.equals(startingPlayer.getLobbyPlayer().getName()))
                    ? "A" : "B";

            outcomes.add(new GameOutcome(
                    winner, playFirst,
                    collector.getCards('A'), collector.getCards('B')));
        }

        int durationSeconds = (int) ((System.currentTimeMillis() - startMillis) / 1000);
        return new PlayedMatch(outcomes, durationSeconds);
    }
}
