package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a written forced resolution's fork and a written damage-step probe's
 * branch could not file is counted in the live game's dropped events, under
 * {@code intervention:<phase>} and {@code probe:<phase>}; a refused or
 * unwritten one adds nothing.
 *
 * <p>No real event reaches no record in either fork: the bracket files what a
 * forced resolution does (a 60-card sweep of spells, with deaths, payments and
 * lost games, counted nothing), and a probe's sink files every event it hears,
 * its stat baseline being taken fresh. So the counts these tests hand over
 * are built, and the merge is what is tested.
 */
@ExtendWith(ForgeExtension.class)
class ForkDroppedEventsKindsTest {

    @TempDir
    Path tempDir;

    private static DroppedEvents oneDrop() {
        DroppedEvents dropped = new DroppedEvents();
        dropped.count(EventType.TAPPED, "main1");
        return dropped;
    }

    private static Map<DroppedEvents.Key, Long> expected(String kind) {
        return Map.of(new DroppedEvents.Key("tapped", kind + ":main1"), 1L);
    }

    @Test
    void aWrittenForcedResolutionsDroppedEventsAreCountedUnderItsKind() throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(2).probes(0).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording live = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 1L, null)) {
            new InterventionFork(game, live, new ForkBudget(caps), 1L).write(List.of(), oneDrop());

            assertEquals(expected("intervention"), live.droppedEvents().counts());
        } finally {
            shard.close();
        }
    }

    @Test
    void aWrittenProbesDroppedEventsAreCountedUnderItsKind() throws Exception {
        assertEquals(expected("probe"), countsAfterWriting(0));
    }

    /** A branch of another turn is not written, and its dropped events are not counted. */
    @Test
    void anUnwrittenProbesDroppedEventsAreNotCounted() throws Exception {
        assertTrue(countsAfterWriting(5).isEmpty());
    }

    /** The live game's counts after a probe branch {@code turnsLate} turns old is offered for writing. */
    private Map<DroppedEvents.Key, Long> countsAfterWriting(int turnsLate) throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(0).probes(1).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording live = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 1L, null)) {
            DamageStepProbe probe = new DamageStepProbe(game, TestCollectors.context(shard, ShardRun.GAME_ID),
                    new ForkBudget(caps), live::countForkDropped);
            boolean written = probe.writeHeldProbe(branch(game.getPhaseHandler().getTurn() + turnsLate), "run.0-l1.9");
            assertEquals(turnsLate == 0, written);
            return Map.copyOf(live.droppedEvents().counts());
        } finally {
            shard.close();
        }
    }

    /** A forced resolution that was refused wrote nothing, so no count reaches the game. */
    @Test
    void aRefusedForcedResolutionAddsNothing() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(2).probes(0).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording live = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 1L, null)) {
            DiscardedCopies.expectOne("intervention", () -> assertFalse(
                    new InterventionFork(null, live, new ForkBudget(caps), 1L).intervene(null, 1)));

            assertTrue(live.droppedEvents().counts().isEmpty(), () -> "" + live.droppedEvents().counts());
        } finally {
            shard.close();
        }
    }

    private static DamageStepProbe.HeldProbe branch(int turn) {
        return new DamageStepProbe.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", CombatShape.of(null), DamageStep.REGULAR, turn,
                Map.of(), oneDrop());
    }
}
