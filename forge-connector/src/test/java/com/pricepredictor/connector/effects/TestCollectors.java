package com.pricepredictor.connector.effects;

import java.nio.file.Path;
import java.util.List;

/**
 * Patched collectors built the two ways a test drives them by hand, and a
 * trait to hand them.
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
     * A stand-in for a trait whose only readable property is its mode.
     *
     * <p>The collector reads the mode reflectively, so anything with the
     * accessor exercises the same path. Using one keeps a test about the
     * parameter reading rather than about which Forge trait happens to declare
     * a given mode.
     */
    public record FakeTrait(String mode) {
        public String getMode() {
            return mode;
        }
    }
}
