package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.GameRecording;
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
import java.util.Objects;
import java.util.Optional;

/**
 * Plays a sealed match between two decks using Forge's AI and reports per-game detail.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class GamePlayer {

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
     * The collection caps, read once: the supervisor sets them when it spawns
     * the worker and they do not change under it, and a malformed value is
     * reported once rather than on every match.
     */
    private final CollectionCaps caps;

    /**
     * Per-game outcome: which side won, which side was on the play, and the
     * non-basic, non-token card names that each side played during that game
     * (controlled by the side that owns them, paperCard names, multiplicities
     * preserved).
     */
    public record GameOutcome(
            Side winner,
            Side playFirst,
            List<String> cardsPlayedA,
            List<String> cardsPlayedB) {
        public GameOutcome {
            Objects.requireNonNull(winner, "winner");
            Objects.requireNonNull(playFirst, "playFirst");
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

        /** Per-game winner sequence, e.g. {@code "ABA"}: the {@code games} field of a match row. */
        public String winnerSequence() {
            return sequence(GameOutcome::winner);
        }

        /** Per-game play-first sequence, same length as {@link #winnerSequence()}. */
        public String playSequence() {
            return sequence(GameOutcome::playFirst);
        }

        private String sequence(java.util.function.Function<GameOutcome, Side> sideOf) {
            StringBuilder sb = new StringBuilder(games.size());
            for (GameOutcome game : games) {
                sb.append(sideOf.apply(game));
            }
            return sb.toString();
        }

        /** Games deck A won. */
        public int winsA() {
            return (int) games.stream().filter(g -> g.winner() == Side.A).count();
        }

        /** Games deck B won. */
        public int winsB() {
            return games.size() - winsA();
        }
    }

    /** Create a player with a configurable best-of-K match format. */
    public GamePlayer(int gamesPerMatch) {
        this(gamesPerMatch, null, CollectionCaps.fromSystemProperties());
    }

    /**
     * Create a player that also collects effect records into {@code writer}.
     *
     * <p>The instrumentation costs no extra simulation: it rides matches that
     * were going to be played anyway.
     */
    public GamePlayer(int gamesPerMatch, RecordShardWriter effectRecords, CollectionCaps caps) {
        this.gamesPerMatch = gamesPerMatch;
        this.effectRecords = effectRecords;
        this.caps = caps;
    }

    /**
     * The two AI-controlled seats a match registers, one of them possibly the
     * random seat (FR-021).
     *
     * <p>Both {@link LobbyPlayerAi} instances are built with a null
     * {@code Set<AIOption>}, which matters beyond style: {@code AiController}
     * only enters full- or hybrid-simulation mode when an {@code AIOption} is
     * set, and every effect-record listener a {@link GameRecording} installs
     * is a plain JVM-wide static with no
     * {@code Game} reference. If simulation ever ran during collection, the
     * hooks could not tell a hypothetical resolution — replayed against a
     * {@code GameCopier} clone purely to evaluate it — from a real one, and
     * hypothetical events would enter the corpus indistinguishable from real
     * ones. {@code GamePlayerTest} pins this against what this method
     * actually builds, not a re-declaration of the constant, because that is
     * the only thing that would notice a future edit passing an
     * {@code AIOption} here.
     *
     * <p>The random seat is a {@link RandomSeatLobbyPlayer}, which is a
     * {@code LobbyPlayerAi} built with the same null option set as the
     * ordinary seats, so the simulation-mode guard holds for it too.
     *
     * <p>Package-private so that test can call it directly rather than
     * playing a match.
     *
     * @param seeds the match's seeded source the random seat's controllers
     *              draw their own seeds from; unused when the seat is none
     */
    static List<RegisteredPlayer> registeredPlayers(
            Deck deckA, Deck deckB, Optional<Side> randomSeat, double probability,
            java.util.Random seeds) {
        return List.of(
                new RegisteredPlayer(deckA).setPlayer(
                        seat(Side.A, randomSeat, probability, seeds)),
                new RegisteredPlayer(deckB).setPlayer(
                        seat(Side.B, randomSeat, probability, seeds))
        );
    }

    private static LobbyPlayerAi seat(
            Side side, Optional<Side> randomSeat, double probability, java.util.Random seeds) {
        return randomSeat.equals(Optional.of(side))
                ? new RandomSeatLobbyPlayer(side.lobbyName(), probability, seeds)
                : new LobbyPlayerAi(side.lobbyName(), null);
    }

    /**
     * Which seat this match gives to the random seat, if any, drawn per
     * match: the share decides whether, a coin decides which (FR-021).
     */
    static Optional<Side> drawRandomSeat(
            CollectionCaps caps, java.util.Random matchRandom) {
        if (!caps.randomSeatEnabled() || matchRandom.nextDouble() >= caps.randomSeatShare()) {
            return Optional.empty();
        }
        return Optional.of(matchRandom.nextBoolean() ? Side.A : Side.B);
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
        // The match's own seeded source, scrambled for the reason the
        // collectors' sampler is: consecutive matches start milliseconds
        // apart, and a linear congruential generator's first draw is nearly
        // a linear function of its seed.
        java.util.Random matchRandom =
                new java.util.Random(GameRecording.scramble(startMillis));
        Optional<Side> randomSeat = drawRandomSeat(caps, matchRandom);
        var players = registeredPlayers(
                deckA, deckB, randomSeat, caps.randomSeatProbability(), matchRandom);

        var rules = new GameRules(GameType.Sealed);
        rules.setGamesPerMatch(gamesPerMatch);

        var match = new Match(rules, players, "sealed-match");

        List<GameOutcome> outcomes = new ArrayList<>();
        // Seeds each game's recording, so a worker's games sample
        // independently of one another rather than all alike.
        long gameSeed = startMillis;

        while (!match.isMatchOver()) {
            playGame(match, gameSeed++).ifPresent(outcomes::add);
        }

        int durationSeconds = (int) ((System.currentTimeMillis() - startMillis) / 1000);
        return new PlayedMatch(outcomes, durationSeconds, randomSeat.isPresent());
    }

    /**
     * Play the match's next game, recording its effects when the run collects
     * them.
     *
     * @param gameSeed seeds the game's recording
     * @return the game's outcome, or empty for a draw or an aborted game
     */
    private Optional<GameOutcome> playGame(Match match, long gameSeed) {
        var game = match.createGame();
        var collector = new PlayedCardCollector(game);
        game.subscribeToEvents(collector);
        // The seat exists before the recording does: the match creates the
        // player, and only then is there a game to record. Found by its
        // controller rather than by lobby name, because the lobby player is
        // what a fork clones and the controller is what a fork does not
        // attach.
        RandomSeatController seat = randomSeatOf(game);
        GameRecording recording = effectRecords == null ? null : new GameRecording(
                game, effectRecords, effectRecords.nextGameId(), caps, gameSeed,
                seat == null ? null : SnapshotBuilder.playerId(seat.getPlayer()));
        try {
            // Inside the try: open() installs JVM-static hooks one by one, and
            // a hook left behind by an open() that threw part way would hand
            // the next game's callbacks to this game's collectors.
            if (recording != null) {
                recording.open();
            }
            if (seat != null) {
                // Live from here: a fork's clone of this seat is never attached
                // and plays as the ordinary AI. Without a recording the seat plays
                // at random and reports nothing -- FR-026 keeps that combination
                // out of a run.
                seat.attach(game, recording == null ? null : recording.playability());
            }
            match.startGame(game); // blocks until game is finished
        } finally {
            if (recording != null) {
                recording.close();
            }
        }

        var winningLobby = game.getOutcome().getWinningLobbyPlayer();
        if (winningLobby == null) {
            return Optional.empty();
        }
        // A name neither side registered throws rather than defaulting to B:
        // a game credited to the wrong deck is a corrupt row.
        Side winner = Side.ofLobbyName(winningLobby.getName());
        Side playFirst = Side.ofLobbyName(
                game.getStartingPlayer().getLobbyPlayer().getName());
        return Optional.of(new GameOutcome(
                winner, playFirst,
                collector.getCards(Side.A), collector.getCards(Side.B)));
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
