package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.phase.PhaseType;
import forge.game.card.Card;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
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
                TestCaps.caps().everyDecision().probeKeywords("trample").build();
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

    // ── collect() over a real combat ────────────────────────────────────

    /** A scheduler over {@code game}'s combat, probing {@code keyword}, its forks real copies. */
    private static ProbeScheduler over(Game game, RecordShardWriter writer, String keyword) {
        CollectionCaps caps = TestCaps.caps().everyDecision().interventions(0).probes(4)
                .probeKeywords(keyword).build();
        return new ProbeScheduler(game, ShardRun.GAME_ID, caps,
                new DamageStepProbe(game, TestCollectors.context(writer, ShardRun.GAME_ID), new ForkBudget(caps)));
    }

    /** {@code attacker} attacking into a Hill Giant, the attacker returned. */
    private static Card blockedBears(Game game, String attacker) {
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card attacking = TestBoard.inPlay(game, you, attacker);
        Card blocker = TestBoard.inPlay(game, opponent, "Hill Giant");
        TestBoard.blockedCombat(game, you, opponent, attacking, blocker);
        return attacking;
    }

    /**
     * A first-strike step nobody strikes in skips its probes, and still drops
     * the branches of a step left behind: a regular step's branch whose step
     * wrote no record would otherwise wait for this combat's regular record.
     */
    @Test
    void aFirstStrikeStepNobodyStrikesInStillDropsAStaleBranch() throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        blockedBears(game, "Grizzly Bears");
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = over(game, writer, "trample");
            probes.hold(branch(DamageStep.REGULAR));

            probes.collect();
            probes.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR);
        }

        assertFalse(Files.exists(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)));
    }

    /**
     * A creature carrying one keyword in two spellings (trample, and trample
     * over planeswalkers) is probed once, both stripped: a branch per
     * spelling would strip one and leave the creature trampling with the other.
     */
    @Test
    void aKeywordACreatureCarriesTwiceIsProbedOnceWithBothGone() throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Card attacker = blockedBears(game, "Colossal Dreadmaw");
        attacker.addChangedCardKeywords(
                List.of("Trample:Planeswalker"), null, false, game.getNextTimestamp(), null);
        assertEquals(2, KeywordEdit.carried(attacker, "trample").size(), "trample in two spellings");
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = over(game, writer, "trample");

            probes.collect();
            probes.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR);
        }

        List<String> lines = ShardRun.readShard(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(1, lines.size(), lines.toString());
        assertTrue(RecordLines.events(lines.get(0)).stream().noneMatch(event ->
                        "damage_dealt".equals(event.get("type"))
                                && String.valueOf(event.get("subjects")).contains("P1")),
                "the branch still trampled over: " + lines.get(0));
    }

    /**
     * First strike is not probed at the regular step: its creature struck in
     * the first-strike step and deals nothing in this one, with it or without
     * it, so the branch would be the real step again.
     */
    @Test
    void firstStrikeIsNotProbedAtTheRegularStep() throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        blockedBears(game, "Youthful Knight");
        try (RecordShardWriter writer = ShardRun.writer(tempDir)) {
            ProbeScheduler probes = over(game, writer, "first_strike");

            probes.collect();
            probes.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR);
        }

        assertFalse(Files.exists(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)));
    }
}
