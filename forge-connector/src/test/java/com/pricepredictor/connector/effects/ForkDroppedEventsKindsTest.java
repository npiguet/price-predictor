package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
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
 * What a written forced resolution's fork could not file is counted in the live game's dropped events,
 * under {@code intervention:<phase>}; a refused one adds nothing.
 *
 * <p>No real event reaches no record in a forced resolution's fork: the
 * bracket files what a forced resolution does (a sweep of about 100 spells,
 * with deaths, payments and lost games, counted nothing). So the counts these
 * tests hand over are built, and the merge is what is tested.
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

    /**
     * Lines really refused on a real game add nothing: Burnt Offering reads
     * what its cost paid and is refused before the fork is taken; Pacifism,
     * held over a board with no creature, has no target to draw, so its
     * fork is taken, spent and refused.
     */
    @Test
    void aRefusedForcedResolutionAddsNothing() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        List<SpellAbility> refused = List.of(
                TestBoard.spellOf(TestBoard.inHand(game, holder, "Burnt Offering"), holder),
                TestBoard.spellOf(TestBoard.inHand(game, holder, "Pacifism"), holder));
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(2).probes(0).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording live = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 1L, null)) {
            for (SpellAbility line : refused) {
                long before = ForkRefusals.counts().values().stream().mapToLong(Long::longValue).sum();

                assertFalse(new InterventionFork(game, live, new ForkBudget(caps), 1L).intervene(line, 1),
                        line + " is refused");

                assertEquals(before + 1, ForkRefusals.counts().values().stream().mapToLong(Long::longValue).sum(),
                        line + " is counted as a refusal: " + ForkRefusals.counts());
            }
            assertTrue(live.droppedEvents().counts().isEmpty(), () -> "" + live.droppedEvents().counts());
        } finally {
            shard.close();
        }
    }
}
