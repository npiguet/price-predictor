package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** When a held probe branch is written, and against which combat record. */
@ExtendWith(ForgeExtension.class)
class ProbeSchedulerTest {

    @TempDir
    Path tempDir;

    private static DamageStepProbe.HeldProbe branch(DamageStep substep) {
        return new DamageStepProbe.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", CombatShape.of(null), substep,
                DamageStepProbe.TURN_UNKNOWN);
    }

    private static ProbeScheduler probing(RecordShardWriter writer) {
        CollectionCaps probing =
                TestCaps.caps().playabilityRate(1.0).probeKeywords("trample").build();
        return new ProbeScheduler(TestCards.game(), ShardRun.GAME_ID, probing,
                new DamageStepProbe(null, TestCollectors.context(writer, ShardRun.GAME_ID), new ForkBudget(probing)));
    }

    /**
     * A branch is completed only against its own damage step.
     *
     * <p>A branch forked at the first-strike step that survived that step and
     * was written against the regular step's record would carry a
     * {@code mirror_of} whose phase is {@code combat_damage}, and gate 2's
     * difference would compare a step that did not happen with one that did.
     */
    @Test
    void aBranchIsCompletedOnlyAgainstItsOwnDamageStep() throws Exception {
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = probing(writer);
            probes.hold(branch(DamageStep.FIRST_STRIKE));
            probes.hold(branch(DamageStep.REGULAR));

            probes.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR);
            // The first-strike branch must still be waiting for its own step.
            probes.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE);
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.10\""), lines.get(1));
    }

    /** With no substep named, the oldest held step is the one being written. */
    @Test
    void theOldestHeldStepIsTheOneACombatRecordCompletes() throws Exception {
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = probing(writer);
            probes.hold(branch(DamageStep.FIRST_STRIKE));
            probes.hold(branch(DamageStep.REGULAR));

            probes.writeHeldProbes("run.0-l1.9");
            // One branch is left, and it is the regular step's.
            probes.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE);
            probes.writeHeldProbes("run.0-l1.11", DamageStep.REGULAR);
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.11\""), lines.get(1));
    }

    /** A branch whose step produced no record is dropped rather than mispaired. */
    @Test
    void aBranchFromAStepThatWroteNoRecordIsDropped() throws Exception {
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = probing(writer);
            probes.hold(branch(DamageStep.FIRST_STRIKE));

            assertEquals(1, probes.discardStaleProbes(DamageStep.REGULAR));
            probes.writeHeldProbes("run.0-l1.9", DamageStep.FIRST_STRIKE);
        }

        assertFalse(Files.exists(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)));
    }
}
