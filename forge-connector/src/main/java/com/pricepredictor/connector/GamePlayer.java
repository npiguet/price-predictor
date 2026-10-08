package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.BusBracketCollector;
import com.pricepredictor.connector.effects.ForkCollector;
import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.SnapshotBuilder;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.Player;
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

    /**
     * Full match outcome returned by {@link #playMatch}.
     *
     * @param randomSeat whether one seat was the random seat (FR-021), in
     *                   which case the match is effect records only and
     *                   neither sealed corpus may take a row from it (FR-025)
     */
    public record PlayedMatch(List<GameOutcome> games, int durationSeconds, boolean randomSeat) {

        /** An ordinary two-AI match. */
        public PlayedMatch(List<GameOutcome> games, int durationSeconds) {
            this(games, durationSeconds, false);
        }
    }

    /** Which seat, if any, the match's random seat takes. */
    enum RandomSeat { NONE, A, B }

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
     * The two AI-controlled seats a match registers.
     *
     * <p>Both {@link LobbyPlayerAi} instances are built with a null
     * {@code Set<AIOption>}, which matters beyond style: {@code AiController}
     * only enters full- or hybrid-simulation mode when an {@code AIOption} is
     * set, and every effect-record listener installed by
     * {@code PatchedCollectors} is a plain JVM-wide static with no
     * {@code Game} reference. If simulation ever ran during collection, the
     * hooks could not tell a hypothetical resolution — replayed against a
     * {@code GameCopier} clone purely to evaluate it — from a real one, and
     * hypothetical events would enter the corpus indistinguishable from real
     * ones. {@code GamePlayerTest} pins this against what this method
     * actually builds, not a re-declaration of the constant, because that is
     * the only thing that would notice a future edit passing an
     * {@code AIOption} here.
     *
     * <p>Package-private so that test can call it directly rather than
     * playing a match.
     */
    static List<RegisteredPlayer> registeredPlayers(Deck deckA, Deck deckB) {
        return registeredPlayers(deckA, deckB, RandomSeat.NONE, 0.0, null);
    }

    /**
     * The two seats, one of them possibly the random seat (FR-021).
     *
     * <p>The random seat is a {@link RandomSeatLobbyPlayer}, which is a
     * {@code LobbyPlayerAi} built with the same null option set as the
     * ordinary seats, so the simulation-mode guard above holds for it too.
     *
     * @param seeds the match's seeded source the random seat's controllers
     *              draw their own seeds from; unused when the seat is none
     */
    static List<RegisteredPlayer> registeredPlayers(
            Deck deckA, Deck deckB, RandomSeat randomSeat, double probability,
            java.util.Random seeds) {
        LobbyPlayerAi seatA = randomSeat == RandomSeat.A
                ? new RandomSeatLobbyPlayer(LOBBY_NAME_A, probability, seeds)
                : new LobbyPlayerAi(LOBBY_NAME_A, null);
        LobbyPlayerAi seatB = randomSeat == RandomSeat.B
                ? new RandomSeatLobbyPlayer(LOBBY_NAME_B, probability, seeds)
                : new LobbyPlayerAi(LOBBY_NAME_B, null);
        return List.of(
                new RegisteredPlayer(deckA).setPlayer(seatA),
                new RegisteredPlayer(deckB).setPlayer(seatB)
        );
    }

    /**
     * Which seat this match gives to the random seat, drawn per match: the
     * share decides whether, a coin decides which (FR-021).
     */
    static RandomSeat drawRandomSeat(
            PatchedCollectors.CollectionCaps caps, java.util.Random matchRandom) {
        if (!caps.randomSeatEnabled() || matchRandom.nextDouble() >= caps.randomSeatShare()) {
            return RandomSeat.NONE;
        }
        return matchRandom.nextBoolean() ? RandomSeat.A : RandomSeat.B;
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
        long startMillis = System.currentTimeMillis();
        // Read once per match rather than per game: the supervisor sets them
        // when it spawns the worker and they do not change under it.
        PatchedCollectors.CollectionCaps caps =
                PatchedCollectors.CollectionCaps.fromSystemProperties();
        // The match's own seeded source, scrambled for the reason the
        // collectors' sampler is: consecutive matches start milliseconds
        // apart, and a linear congruential generator's first draw is nearly
        // a linear function of its seed.
        java.util.Random matchRandom =
                new java.util.Random(PatchedCollectors.scramble(startMillis));
        RandomSeat randomSeat = drawRandomSeat(caps, matchRandom);
        var players = registeredPlayers(
                deckA, deckB, randomSeat, caps.randomSeatProbability(), matchRandom);

        var rules = new GameRules(GameType.Sealed);
        rules.setGamesPerMatch(gamesPerMatch);

        var match = new Match(rules, players, "sealed-match");

        List<GameOutcome> outcomes = new ArrayList<>();
        // Seeds the patched collectors' sampling, so a worker's games sample
        // independently of one another rather than all alike.
        long gameSeed = startMillis;

        while (!match.isMatchOver()) {
            var game = match.createGame();
            var collector = new PlayedCardCollector();
            game.subscribeToEvents(collector);
            PatchedCollectors patched = null;
            // The seat exists before the collectors do: the match creates the
            // player, and only then is there a game to collect from. Found by
            // its controller rather than by lobby name, because the lobby
            // player is what a fork clones and the controller is what a fork
            // does not attach.
            RandomSeatController seat = randomSeatOf(game);
            String randomSeatPlayerId = seat == null
                    ? null : SnapshotBuilder.playerId(seat.getPlayer());
            if (effectRecords != null) {
                // One collector per game, with the game's own id: records join
                // to a checkpoint's recorded split by game_id, so the id has to
                // change when the game does. Both collectors share that id —
                // they describe the same game.
                String gameId = effectRecords.nextGameId();
                BusBracketCollector bracket = new BusBracketCollector(
                        game, effectRecords, gameId);
                bracket.setRandomSeatPlayerId(randomSeatPlayerId);
                game.subscribeToEvents(bracket);
                // The patched collectors reach the engine through static
                // listeners, so they are installed for the life of one game and
                // uninstalled after it. On a stock checkout install() finds no
                // hook and the game plays exactly as it did before.
                patched = new PatchedCollectors(
                        game, effectRecords, gameId, caps, gameSeed);
                patched.setRandomSeatPlayerId(randomSeatPlayerId);
                // Where an effect API's own parameters describe its outcome,
                // the clause hook files the event through the same bracket
                // the bus-derived events land in.
                patched.withBracket(bracket);
                if (caps.interventionsPerGame() > 0 || caps.probesEnabled()) {
                    // Stage three. Nothing is copied unless a budget says so,
                    // because a fork costs a game copy and a stack resolution.
                    patched.withForks(new ForkCollector(
                            game, effectRecords, gameId, caps, gameSeed));
                }
                gameSeed++;
                // A probe forks before the damage step and cannot know which
                // combat record it mirrors until that record is written.
                final PatchedCollectors withProbes = patched;
                bracket.onCombatRecord(withProbes::writeHeldProbes);
                patched.install();
                // Continuous effects are read off the layer tables at a phase
                // boundary, which is the cheapest moment the board is stable.
                game.subscribeToEvents(patched.phaseBridge());
            }
            if (seat != null) {
                // Live from here: a fork's clone of this seat is never
                // attached and plays as the ordinary AI. The collectors may be
                // null, in which case the seat plays at random and reports
                // nothing -- FR-026 keeps that combination out of a run.
                seat.attach(game, patched);
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
        return new PlayedMatch(outcomes, durationSeconds, randomSeat != RandomSeat.NONE);
    }

    /** The game's random seat, by its controller, or null when it has none. */
    static RandomSeatController randomSeatOf(forge.game.Game game) {
        for (Player player : game.getPlayers()) {
            if (player.getController() instanceof RandomSeatController seat) {
                return seat;
            }
        }
        return null;
    }
}
