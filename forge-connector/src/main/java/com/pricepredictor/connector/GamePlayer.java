package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.DeckRules;
import com.pricepredictor.connector.effects.FruitlessActivations;
import com.pricepredictor.connector.effects.GameRecording;
import com.pricepredictor.connector.effects.SeedStreams;
import com.pricepredictor.connector.effects.PriorityCandidates;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.Refs;
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

    /** Asked at each game end: whether the worker is to stop there (D2). */
    private final WorkerStop stop;

    /**
     * The games this player has played to their end, recording closed: a
     * stopped worker's stop line names them, since a best-of-N match stopped
     * after a game counts as no match.
     */
    private long gamesPlayed;

    /** Plays one game of a match to its end; {@link Match#startGame} but in a test. */
    private final GameDriver driver;

    /** The match's start in milliseconds, which seeds it; the wall clock but in a test. */
    private final java.util.function.LongSupplier clock;

    /** How a game of a match is played to its end, a seam for a test that makes one throw. */
    @FunctionalInterface
    interface GameDriver {
        void play(Match match, forge.game.Game game);
    }

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

        /** The game's winner and starter, without the cards it played. */
        public GameResult result() {
            return new GameResult(winner, playFirst);
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

        /** Each game's winner and starter, in game order: what a match row records of it. */
        public List<GameResult> results() {
            return games.stream().map(GameOutcome::result).toList();
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
        this(gamesPerMatch, effectRecords, caps, WorkerStop.NEVER, Match::startGame);
    }

    /**
     * A player that also stops at a game end when {@code stop} asks it to,
     * throwing {@link WorkerStop.Requested} out of {@link #playMatch} once the
     * game's recording is closed and its marker written.
     */
    GamePlayer(int gamesPerMatch, RecordShardWriter effectRecords, CollectionCaps caps,
            WorkerStop stop, GameDriver driver) {
        this(gamesPerMatch, effectRecords, caps, stop, driver, System::currentTimeMillis);
    }

    /**
     * The same, the match's start read from {@code clock}: a test replaying
     * one match twice gives both the same start, which seeds the match's
     * source and its games' recordings.
     */
    GamePlayer(int gamesPerMatch, RecordShardWriter effectRecords, CollectionCaps caps,
            WorkerStop stop, GameDriver driver, java.util.function.LongSupplier clock) {
        this.gamesPerMatch = gamesPerMatch;
        this.effectRecords = effectRecords;
        this.caps = caps;
        this.stop = stop;
        this.driver = driver;
        this.clock = clock;
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
        return playMatch(deckA, deckB, DeckRules.NONE);
    }

    /** The seed of a match started at {@code startMillis}: the start mixed with this worker lifetime's token. */
    private long matchSeed(long startMillis) {
        return SeedStreams.matchSeed(startMillis, effectRecords == null ? "" : effectRecords.token());
    }

    /** The games played to their end so far, in matches finished or not. */
    public long gamesPlayed() {
        return gamesPlayed;
    }

    /**
     * {@link #playMatch(Deck, Deck)} for decks aimed at {@code deckRules}: a
     * steered variant round's decks (FR-058d), whose games' records carry
     * both decks' rules as {@code aimed_at}.
     */
    public PlayedMatch playMatch(Deck deckA, Deck deckB, DeckRules deckRules) {
        long wallStart = System.currentTimeMillis();
        long startMillis = clock.getAsLong();
        // The match's own seeded source: the seat consumer's stream of the
        // match's first game's seed, which no other consumer shares.
        long matchSeed = matchSeed(startMillis);
        java.util.Random matchRandom = SeedStreams.random(matchSeed, SeedStreams.Consumer.SEAT);
        Optional<Side> randomSeat = drawRandomSeat(caps, matchRandom);
        List<GameOutcome> outcomes = new ArrayList<>();
        // Seeds each game's recording, so a worker's games sample
        // independently of one another rather than all alike, and two
        // workers' games never replay one another's streams.
        long gameSeed = matchSeed;

        // From the seat's draw on, whatever fails is a random-seat match's
        // failure when the match has one: setting the match up, playing a
        // game, closing its recording.
        try {
            var players = registeredPlayers(
                    deckA, deckB, randomSeat, caps.randomSeatProbability(), matchRandom);
            var rules = new GameRules(GameType.Sealed);
            rules.setGamesPerMatch(gamesPerMatch);
            var match = new Match(rules, players, "sealed-match");
            while (!match.isMatchOver()) {
                playGame(match, gameSeed++, deckRules).ifPresent(outcomes::add);
                // At the game end, its records and marker written: the match
                // is left unfinished and records no row (D2).
                var reason = stop.reason();
                if (reason.isPresent() && !match.isMatchOver()) {
                    throw new WorkerStop.Requested(reason.get());
                }
            }
        } catch (RandomSeatMatchFailed | WorkerStop.Requested passOn) {
            throw passOn;
        } catch (RuntimeException failure) {
            if (randomSeat.isPresent()) {
                throw new RandomSeatMatchFailed(failure);
            }
            throw failure;
        } catch (Error error) {
            // An Error is never caught: it ends the worker. Named here first,
            // in the worker loop's own words, so a random-seat match it cut is
            // counted as one.
            if (randomSeat.isPresent()) {
                System.err.println("Error generating match: " + RandomSeatMatchFailed.MARK + error);
                System.err.flush();
            }
            throw error;
        }

        int durationSeconds = (int) ((System.currentTimeMillis() - wallStart) / 1000);
        return new PlayedMatch(outcomes, durationSeconds, randomSeat.isPresent());
    }

    /**
     * A random-seat match that threw: its message names it as one, so the
     * worker's {@code Error generating match: [random-seat match] ...} line
     * tells the supervisor which crashed matches were random-seat ones.
     */
    static final class RandomSeatMatchFailed extends RuntimeException {
        /** What the message starts with, after the worker loop's prefix. */
        static final String MARK = "[random-seat match] ";

        RandomSeatMatchFailed(RuntimeException cause) {
            super(MARK + cause.getMessage(), cause);
        }
    }

    /**
     * Play the match's next game, recording its effects when the run collects
     * them.
     *
     * @param gameSeed seeds the game's recording
     * @param rules    the rules the match's decks are aimed at
     * @return the game's outcome, or empty for a draw or an aborted game
     */
    private Optional<GameOutcome> playGame(Match match, long gameSeed, DeckRules rules) {
        var game = match.createGame();
        var collector = new PlayedCardCollector(game);
        game.subscribeToEvents(collector);
        // The seat exists before the recording does: the match creates the
        // player, and only then is there a game to record. Found by its
        // controller rather than by lobby name, because the lobby player is
        // what a fork clones and the controller is what a fork does not
        // attach.
        Optional<RandomSeatController> seat = randomSeatOf(game);
        Throwable thrown = null;
        GameRecording recording = effectRecords == null ? null : new GameRecording(
                game, effectRecords, effectRecords.nextGameId(), caps, gameSeed,
                seat.map(s -> Refs.of(s.getPlayer())).orElse(null), GamePlayer::attachForkSeats,
                rules);
        try {
            // Inside the try: open() installs JVM-static hooks one by one, and
            // a hook left behind by an open() that threw part way would hand
            // the next game's callbacks to this game's collectors.
            if (recording != null) {
                recording.open();
            }
            // Live from here. A fork's clone of this seat plays as the
            // ordinary AI unless the fork is played on, which attaches it
            // (attachForkSeats). Without a recording the seat plays at random
            // and reports nothing -- FR-026 keeps that combination out of a
            // run.
            seat.ifPresent(s -> s.attach(
                    game, recording == null ? null : recording.priorityCandidates(),
                    recording == null ? null : recording.fruitlessActivations()));
            driver.play(match, game); // blocks until game is finished
        } catch (RuntimeException | Error failure) {
            thrown = failure;
            throw failure;
        } finally {
            if (recording != null) {
                closeRecording(recording::close, thrown);
            }
        }
        gamesPlayed++;

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

    /**
     * Close a game's recording; a close that throws after the game threw is
     * kept with the game's failure, as suppressed, rather than masking it:
     * an {@code Error} too (an {@code OutOfMemoryError} in the shard
     * writer), since the game's own failure is what the worker reports.
     *
     * @param close       closes the recording
     * @param gameFailure what the game threw, or null when it ended
     */
    static void closeRecording(Runnable close, Throwable gameFailure) {
        try {
            close.run();
        } catch (RuntimeException | Error closeFailure) {
            if (gameFailure == null) {
                throw closeFailure;
            }
            gameFailure.addSuppressed(closeFailure);
        }
    }

    /**
     * Attach a fork's seats to the fork, as {@link #playGame} attaches the
     * live game's (D4): the random seat with draws seeded from the game's
     * fork seed, so it chooses afresh in the fork as it would live, and
     * nothing drawn from the match's source. Both other seats are the
     * plain AI the copy built.
     */
    public static void attachForkSeats(
            forge.game.Game fork, PriorityCandidates candidates,
            FruitlessActivations fruitless, java.util.Random random) {
        randomSeatOf(fork).ifPresent(s -> s.attach(fork, candidates, fruitless, random));
    }

    /** The game's random seat, by its controller, or empty when it has none. */
    static Optional<RandomSeatController> randomSeatOf(forge.game.Game game) {
        for (Player player : game.getPlayers()) {
            if (player.getController() instanceof RandomSeatController seat) {
                return Optional.of(seat);
            }
        }
        return Optional.empty();
    }
}
