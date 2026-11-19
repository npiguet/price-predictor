package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.Collections;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The independent random streams one game's consumers draw from, all derived
 * from the game's seed.
 *
 * <p>Each consumer has a fixed {@link Consumer} label, and its stream is a
 * {@link SplittableRandom} rooted at the game's seed and that label alone, so
 * the split does not depend on the order consumers are built in: adding a
 * consumer shifts no other's stream. Two consumers of one game, or one
 * consumer of two games, never replay the same draws, whatever the games'
 * seeds (consecutive milliseconds, say): {@link SplittableRandom}'s output is
 * a bijective mix of its state.
 *
 * <p>A fork's own stream is the game's {@link Consumer#FORK_STREAMS} stream
 * advanced by the fork's per-game ordinal ({@link GameStreams}), never a
 * JVM-wide count, so the n-th fork of two games, or of two worker lifetimes,
 * draws differently.
 */
public final class SeedStreams {

    /** The consumers of a game's seed, each with a fixed salt that is never reused. */
    public enum Consumer {
        /** The match's random-seat draw (seeded from the match's first game's seed). */
        SEAT(0x2C1B3C6D0A5E8F17L),
        /** The collectors' sampling stream. */
        SAMPLER(0x297A2D39B4C06E55L),
        /** The intervention fork's line order and choices. */
        INTERVENTION_CHOICES(0x6E1F5A0C92D3B748L),
        /** The intervention spread gate's draws. */
        SPREAD_GATE(0x2545F4914F6CDD1DL),
        /** The directed-fork scheduler's probability draws. */
        DIRECTED_FORKS(0x5DEECE66DA3C8B21L),
        /** The protection fork's probability draws. */
        PROTECTION(0x7F4A7C159E3779B9L),
        /** The seeds a fork played on takes for its seats and its sampling. */
        PLAY_FORWARD_SEEDS(0x3F84D5B5B5470917L),
        /** The streams of the game's forks, advanced by each fork's ordinal. */
        FORK_STREAMS(0x1D8E4E27C47D124FL);

        private final long salt;

        Consumer(long salt) {
            this.salt = salt;
        }
    }

    /** The game's streams, by game, held weakly so a finished game is not kept. */
    private static final Map<Game, GameStreams> BY_GAME =
            Collections.synchronizedMap(new WeakHashMap<>());

    private static final long TOKEN_BITS = (1L << 40) - 1;

    private SeedStreams() {
    }

    /**
     * The seed of a match's first game: its start time, mixed with the token of
     * the worker lifetime that plays it ({@code {worker}-{lifetime}}, as the
     * shard file is named; empty for a player collecting no records).
     *
     * <p>Two workers whose matches start in the same millisecond would share
     * the start time, and so every stream of the game. The token is hashed to
     * 40 bits and XORed into the low bits, so two games' seeds still differ by
     * under 2^42, which keeps the consumers' salts apart (their pairwise XORs
     * have bit length 59 or more). The match's later games add their index to
     * this seed.
     */
    public static long matchSeed(long startMillis, String workerToken) {
        return startMillis ^ (new SplittableRandom(workerToken.hashCode()).split().nextLong()
                & TOKEN_BITS);
    }

    /** The seed of {@code consumer}'s stream in the game seeded {@code gameSeed}. */
    public static long seed(long gameSeed, Consumer consumer) {
        return new SplittableRandom(gameSeed ^ consumer.salt).split().nextLong();
    }

    /** {@code consumer}'s stream in the game seeded {@code gameSeed}. */
    public static Random random(long gameSeed, Consumer consumer) {
        return new Random(seed(gameSeed, consumer));
    }

    /** Record that {@code game} is the game seeded {@code gameSeed}: its forks draw from it. */
    static void register(Game game, long gameSeed) {
        BY_GAME.put(game, new GameStreams(gameSeed));
    }

    /**
     * The next fork's stream of {@code live}: a game no recording seeded is
     * the game seeded 0.
     */
    static Random nextForkStream(Game live) {
        return BY_GAME.computeIfAbsent(live, game -> new GameStreams(0L)).nextFork();
    }

    /** One game's seed and the number of forks it has taken. */
    static final class GameStreams {
        private final long forkRoot;
        private final AtomicLong forks = new AtomicLong();

        GameStreams(long gameSeed) {
            this.forkRoot = seed(gameSeed, Consumer.FORK_STREAMS);
        }

        /** The stream of the game's next fork, numbered from 0. */
        Random nextFork() {
            return new Random(new SplittableRandom(forkRoot + forks.getAndIncrement()).nextLong());
        }
    }
}
