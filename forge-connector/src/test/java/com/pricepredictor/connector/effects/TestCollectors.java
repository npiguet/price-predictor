package com.pricepredictor.connector.effects;

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

    /**
     * A collector over no game and no writer.
     *
     * <p>Enough for what it decides before anything is written: whether a
     * record coalesces, whether a trigger evaluation is offered, which hooks it
     * installs.
     */
    static PatchedCollectors unwired(CollectionCaps caps) {
        return new PatchedCollectors(null, null, "game-1", caps, 42L);
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
        return new PatchedCollectors(
                TestCards.game(),
                new RecordShardWriter(dir, "run", 0, "l1"),
                "run.0-l1.0", CollectionCaps.defaults(), 1L);
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
