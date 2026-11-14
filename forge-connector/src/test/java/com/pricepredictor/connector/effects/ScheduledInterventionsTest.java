package com.pricepredictor.connector.effects;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

/**
 * The intervention scheduler as a worker runs it: a started game recorded by
 * a {@link GameRecording} with every hook installed, its phase boundaries
 * published on the game's bus, and real forks through {@code GameCopier},
 * each boundary drawing against the worker's gate.
 */
@ExtendWith(ForgeExtension.class)
class ScheduledInterventionsTest {

    /**
     * Boundaries published per stretch: at one in twenty, the gate passes
     * at about ten of them, so a line that can be forced is.
     */
    private static final int BOUNDARIES = 200;

    @TempDir
    Path tempDir;

    /** Publish {@code count} boundaries of the step the game is in. */
    private static void boundaries(Game game, int count) {
        for (int i = 0; i < count; i++) {
            game.fireEvent(new GameEventTurnPhase(
                    game.getPhaseHandler().getPlayerTurn(), game.getPhaseHandler().getPhase(), ""));
        }
    }

    /**
     * The forced resolutions' records the recording wrote: interventional
     * resolution halves with no partner, which leaves out a protection fork's.
     */
    private List<String> forced(Game game, int interventions, ShardRun.Played body) throws Throwable {
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(interventions).probes(0).build();
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, game, caps, "to force a line as a worker does", body);
        if (!Files.exists(writer.path())) {
            return List.of();
        }
        return ShardRun.readShard(writer.path()).stream()
                .filter(line -> Boolean.TRUE.equals(parse(line).get("interventional"))
                        && "resolution".equals(parse(line).get("kind"))
                        && parse(line).get("link_id") == null)
                .toList();
    }

    /**
     * A removal spell held over an empty board has nothing to resolve
     * against: it spends no fork and is not marked forced, so once a creature
     * arrives it is forced under the budget of one it would otherwise have
     * spent. The live board is the same before and after the fork.
     */
    @Test
    void aSpellWithNoTargetIsForcedOnceATargetArrives() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inHand(game, you, "Doom Blade");
        Map<String, Object>[] boards = new Map[2];

        List<String> records = forced(game, 1, () -> {
            boundaries(game, BOUNDARIES);
            TestBoard.inPlay(game, opponent, "Grizzly Bears");
            game.getTriggerHandler().resetActiveTriggers();
            boards[0] = BoardStates.of(game);
            boundaries(game, BOUNDARIES);
            boards[1] = BoardStates.of(game);
        });

        assertEquals(1, records.size(), records.toString());
        assertTrue(eventTypes(records.get(0)).contains(EventType.DESTROYED.wireName())
                        || eventTypes(records.get(0)).contains(EventType.ZONE_CHANGE.wireName()),
                records.get(0));
        assertEquals(boards[0], boards[1], "the fork reached the live board");
    }

    /**
     * Two lines of one card with the same effect API (Chainflinger's two
     * damage abilities) are two lines, each forced once.
     */
    @Test
    void twoLinesOfOneCardWithOneApiAreForcedApart() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Chainflinger");

        List<String> records = forced(game, 4, () -> boundaries(game, 2 * BOUNDARIES));

        assertEquals(2, records.size(), records.toString());
        assertNotEquals(parse(records.get(0)).get("ability"), parse(records.get(1)).get("ability"),
                records.toString());
    }

    /**
     * No fork is taken at a phase event published while the stack resolves,
     * as an effect ending the turn publishes the steps it moves to; the
     * line is forced at the next boundary that draws one.
     */
    @Test
    void noForkIsTakenWhileTheStackResolves() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inHand(game, game.getPlayers().get(0), "Lightning Bolt");

        List<String> records = forced(game, 1, () -> {
            game.getStack().setResolving(true);
            try {
                boundaries(game, BOUNDARIES);
            } finally {
                game.getStack().setResolving(false);
            }
            // What tells the boundary the fork was taken at.
            game.getPlayers().get(1).setLife(7, null);
            boundaries(game, BOUNDARIES);
        });

        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("{\"id\":\"P1\",\"life\":7,"),
                "forked while the stack resolved: " + records.get(0));
    }
}
