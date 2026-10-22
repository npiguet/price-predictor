package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementHandler;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * Patched collectors and their channels built the ways a test drives them by
 * hand, and the traits to hand them.
 *
 * <p>A test that needs the bracket wired, or reads the shard back, runs
 * through {@link ShardRun} instead.
 */
final class TestCollectors {

    /**
     * One game's channels, built over one game, one dedup set and one sampling
     * stream as {@link PatchedCollectors} builds them, for a test that calls
     * a channel's handler or record builder by hand.
     *
     * <p>Closing writes what the mana channel holds and ends the game for the
     * rewrite tally, as closing the patched collectors does; no hook is
     * installed, so none comes down.
     */
    record Channels(
            RewriteChannel rewrites,
            TriggerChannel triggers,
            PlayabilityChannel playability,
            ManaChannel mana,
            ClauseOutcomeRouter clauses,
            ContinuousChannel continuous) implements AutoCloseable {

        static Channels of(
                Game game, CollectorContext context, CollectionCaps caps, long seed,
                BusBracketCollector bracket) {
            LiveGame live = new LiveGame(game);
            RecordDedup dedup = new RecordDedup();
            Random sampler = new Random(GameRecording.scramble(seed));
            return new Channels(
                    new RewriteChannel(live, context, dedup),
                    new TriggerChannel(live, context, dedup, new TriggerNegativeSampler(sampler)),
                    new PlayabilityChannel(live, context, dedup, sampler, caps.playabilityRate()),
                    new ManaChannel(live, context, new ManaReservoir(caps.manaCap(), sampler)),
                    new ClauseOutcomeRouter(live, bracket),
                    new ContinuousChannel(live, context, dedup));
        }

        @Override
        public void close() {
            try {
                mana.flush();
            } finally {
                RewriteChannel.JVM_REWRITES.endOfGame();
            }
        }
    }

    private TestCollectors() {
    }

    /** The two caps these tests vary; two interventions and two probes, no keyword. */
    static CollectionCaps caps(int manaCap, double playabilityRate) {
        return new CollectionCaps(
                manaCap, playabilityRate, 2, 2, List.of(), 0.0, 0.0);
    }

    /** A collector's context for a match with no random seat, in this checkout's mode. */
    static CollectorContext context(RecordShardWriter writer, String gameId) {
        return context(writer, gameId, null);
    }

    /** The same, for a match whose random seat is {@code randomSeatPlayerId}. */
    static CollectorContext context(
            RecordShardWriter writer, String gameId, String randomSeatPlayerId) {
        return new CollectorContext(writer, gameId, AttributionMode.detect(), randomSeatPlayerId);
    }

    /** A bracket writing to {@code writer}, which nothing is told a combat record of. */
    static BusBracketCollector bracket(Game game, RecordShardWriter writer, String gameId) {
        return new BusBracketCollector(game, context(writer, gameId), recordId -> { });
    }

    /**
     * A collector over no game and no writer.
     *
     * <p>Enough for what it decides before anything is written: whether a
     * record coalesces, whether a trigger evaluation is offered, which hooks it
     * installs.
     */
    static PatchedCollectors unwired(CollectionCaps caps) {
        return new PatchedCollectors(
                null, context(null, "game-1"), caps, 42L, bracket(null, null, "game-1"));
    }

    /**
     * Channels over a game nothing is played in.
     *
     * <p>Enough for a snapshot and a record id, which is all a record builder
     * needs: what is under test is what the channel reads off the trait the
     * hook hands it, and that read is the same whether or not a turn has been
     * taken.
     */
    static Channels recording(Path dir) {
        Game game = TestCards.game();
        RecordShardWriter writer = new RecordShardWriter(dir, "run", 0, "l1");
        return Channels.of(
                game, context(writer, "run.0-l1.0"), CollectionCaps.defaults(), 1L,
                bracket(game, writer, "run.0-l1.0"));
    }

    /**
     * A trigger of a chosen mode, on a host card the mode does not care about.
     *
     * <p>Parsed rather than taken off a real card, because what is under test
     * is the reading of one mode's run parameters and finding a printed card
     * per mode would tie the test to the set list.
     */
    static Trigger triggerOfMode(String mode) {
        return TriggerHandler.parseTrigger(
                "Mode$ " + mode + " | TriggerZones$ Battlefield"
                        + " | TriggerDescription$ under test",
                TestCards.build("Paralyze"), true);
    }

    /** A replacement of a chosen mode, parsed for the reason {@link #triggerOfMode} is. */
    static ReplacementEffect replacementOfMode(String mode) {
        return ReplacementHandler.parseReplacement(
                "Event$ " + mode + " | Description$ under test",
                TestCards.build("Paralyze"), true);
    }
}
