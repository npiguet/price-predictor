package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.DirectedRules;
import forge.game.Game;

import java.util.List;

/**
 * What one game's {@link GameRecording} is built from.
 *
 * <p>A request starts from {@link #of} (a match that needs no seats attached
 * in a fork, decks aimed at no rule, the registered directed rules and the
 * JVM lifetime's tally, no random seat) and each {@code with} method sets the
 * one thing a run adds.
 *
 * @param game               the game recorded
 * @param writer             where its records go
 * @param gameId             the game's own id: records join to a
 *                           checkpoint's recorded split by it, so it changes
 *                           when the game does
 * @param caps               the run's collection caps
 * @param seed               seeds the sampling and the forks' choices, so a
 *                           worker's games sample independently of one
 *                           another
 * @param randomSeatPlayerId the match's random seat, as a player id, or null
 *                           when it has none
 * @param forkSeats          attaches a fork's seats when a fork of this game
 *                           plays on
 * @param decks              the rules the game's decks are aimed at
 *                           (FR-058d): when either carries a rule, the game
 *                           is steered, and every record it writes carries
 *                           both decks' rules as {@code aimed_at} instead of
 *                           the run's list
 * @param directedRules      the rules directed forks are taken for
 * @param tally              where directed forks are counted
 */
public record GameRecordingRequest(
        Game game, RecordShardWriter writer, String gameId, CollectionCaps caps, long seed,
        String randomSeatPlayerId, ForkSeats forkSeats, DeckRules decks,
        DirectedRules directedRules, DirectedForkTally tally) {

    /**
     * A request for {@code game} written to {@code writer} under {@code gameId}, with
     * seed 0: {@link #seeded} gives a worker's games their own.
     */
    public static GameRecordingRequest of(
            Game game, RecordShardWriter writer, String gameId, CollectionCaps caps) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, 0L, null, ForkSeats.NONE, DeckRules.NONE,
                DirectedRules.registered(), DirectedForkTally.lifetime());
    }

    /** This request with {@code newSeed} seeding its sampling and its forks' choices. */
    public GameRecordingRequest seeded(long newSeed) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, newSeed, randomSeatPlayerId, forkSeats, decks,
                directedRules, tally);
    }

    /** This request with the match's random seat, as a player id, or null for none. */
    public GameRecordingRequest withRandomSeat(String playerId) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, seed, playerId, forkSeats, decks, directedRules, tally);
    }

    /** This request with {@code seats} attaching a fork's seats. */
    public GameRecordingRequest withForkSeats(ForkSeats seats) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, seed, randomSeatPlayerId, seats, decks,
                directedRules, tally);
    }

    /** This request for a game whose decks are aimed at {@code aimedDecks}. */
    public GameRecordingRequest withDecks(DeckRules aimedDecks) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, seed, randomSeatPlayerId, forkSeats, aimedDecks,
                directedRules, tally);
    }

    /** This request with directed forks taken for {@code rules} and counted in {@code counts}. */
    GameRecordingRequest withDirected(DirectedRules rules, DirectedForkTally counts) {
        return new GameRecordingRequest(
                game, writer, gameId, caps, seed, randomSeatPlayerId, forkSeats, decks,
                rules, counts);
    }

    /** The game's aim: the run's list, or a steered game's decks' rules. */
    List<String> aimedAt() {
        return decks.aimedAt(caps.aimedAt());
    }

    /**
     * A context of this game for one collector, in {@code mode}. Each
     * collector has its own, because each notes whether a record it wrote
     * held the game's outcome ({@link CollectorContext#wroteTheOutcome}).
     */
    CollectorContext newContext(AttributionMode mode) {
        return new CollectorContext(writer, gameId, mode, randomSeatPlayerId, aimedAt());
    }
}
