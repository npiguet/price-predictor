package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementHandler;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;

import java.nio.file.Path;
import java.util.List;

/**
 * Patched collectors built the two ways a test drives them by hand, and the
 * traits to hand them.
 *
 * <p>A test that needs the bracket wired, or reads the shard back, runs
 * through {@link ShardRun} instead.
 */
final class TestCollectors {

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
        return new BusBracketCollector(game, context(writer, gameId));
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
                null, context(null, "game-1"), caps, 42L, bracket(null, null, "game-1"), null);
    }

    /**
     * A collector over a game nothing is played in.
     *
     * <p>Enough for a snapshot and a record id, which is all a record builder
     * needs: what is under test is what the collector reads off the trait the
     * hook hands it, and that read is the same whether or not a turn has been
     * taken.
     */
    static PatchedCollectors recording(Path dir) {
        Game game = TestCards.game();
        RecordShardWriter writer = new RecordShardWriter(dir, "run", 0, "l1");
        return new PatchedCollectors(
                game, context(writer, "run.0-l1.0"), CollectionCaps.defaults(), 1L,
                bracket(game, writer, "run.0-l1.0"), null);
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
