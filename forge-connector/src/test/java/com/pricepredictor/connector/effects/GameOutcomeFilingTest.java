package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.withEventOfType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Where the game's outcome is written (findings 9.5, 9.6, 9.13): {@code
 * player_lost} and {@code player_won} never join an open cast's bracket; the
 * game-end record holds those two and nothing else, every other orphan alive
 * at the end counted as dropped; a spell that ends the game keeps its card's
 * move off the stack in its own half; and the marker follows only a written
 * outcome.
 */
@ExtendWith(ForgeExtension.class)
class GameOutcomeFilingTest {

    private static final String WHY = "for the game-end marker's recording";

    @TempDir
    Path tempDir;

    /**
     * Mischievous Poltergeist's "Pay 1 life" activated at 1 life, as the
     * pilot's random seat did: the state-based check after the activation
     * ends the game with the ability still on the stack. The loss and the win
     * are each written once, in the game-end record, and the marker follows.
     */
    @Test
    void aLifeCostPaidToZeroWritesTheOutcomeOnceAndTheMarkerAfterIt() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player payer = game.getPlayers().get(0);
        Player winner = game.getPlayers().get(1);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            payLifeToZero(game, payer);
            game.fireEvent(new GameEventGameFinished());
        });

        List<String> lost = withEventOfType(records, "player_lost");
        assertEquals(1, lost.size(), records.toString());
        assertEquals(lost, withEventOfType(records, "player_won"), records.toString());
        String end = lost.get(0);
        assertEquals(List.of("player_lost", "player_won"), eventTypes(end), end);
        assertEquals(List.of(Refs.of(payer)), events(end).get(0).get("subjects"), end);
        assertEquals("LifeReachedZero", params(events(end).get(0)).get("reason"), end);
        assertEquals(List.of(Refs.of(winner)), events(end).get(1).get("subjects"), end);
        assertTrue(isMarker(records.get(records.size() - 1)), "the marker last: " + records);
        assertTrue(records.indexOf(end) < records.size() - 1, records.toString());
    }

    /**
     * The same game, recorded without the game's end ever being published:
     * its outcome reaches no record, so the game writes no marker and reads
     * as cut rather than as a game with no result.
     */
    @Test
    void aGameWhoseOutcomeWasNeverWrittenHasNoMarker() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player payer = game.getPlayers().get(0);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, WHY, () -> payLifeToZero(game, payer));

        assertEquals(List.of(), withEventOfType(records, "player_lost"), records.toString());
        assertFalse(records.stream().anyMatch(GameOutcomeFilingTest::isMarker), records.toString());
    }

    /**
     * Forest played, Juggernaut attacks for 5, and the next turn the
     * defender draws from an empty library: the game ends with the land's
     * move and the attack's tap still orphaned. The game-end record holds
     * the loss and the win only; the move and the tap are counted as dropped.
     */
    @Test
    void theGameEndRecordHoldsTheOutcomeOnlyAndTheRestIsDropped() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player attacking = game.getPlayers().get(0);
        Card forest = TestBoard.inHand(game, attacking, "Forest");
        TestBoard.inPlay(game, attacking, "Juggernaut");
        AtomicReference<Map<DroppedEvents.Key, Long>> dropped = new AtomicReference<>();

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            attacking.playLand(forest, null);
            playUntilTheGameEnds(game);
            game.fireEvent(new GameEventGameFinished());
            dropped.set(Map.copyOf(collector.droppedEvents().counts()));
        });

        String end = only(withEventOfType(records, "player_lost"));
        assertEquals(List.of("player_lost", "player_won"), eventTypes(end), end);
        assertEquals("Milled", params(events(end).get(0)).get("reason"), end);
        assertEquals(1L, dropped.get().get(new DroppedEvents.Key("zone_change", "main1")),
                "the land's move: " + dropped.get());
        assertEquals(1L, dropped.get().get(new DroppedEvents.Key("tapped", "combat_declare_attackers")),
                "the attack's tap: " + dropped.get());
    }

    /**
     * One with Death ends the game as it resolves: its effect half holds the
     * loss, the win and then its card's move off the stack, and the game
     * writes no game-end record.
     */
    @Test
    void aSpellThatEndsTheGameKeepsItsCardsMoveInItsOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            TestBoard.castUntargeted(game, caster, "One with Death");
            game.getStack().resolveStack();
            assertTrue(game.isGameOver(), "the caster lost");
            game.fireEvent(new GameEventGameFinished());
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/o/one_with_death.txt");
        assertEquals(List.of("player_lost", "player_won", "zone_change"), eventTypes(half), half);
        assertEquals("stack", params(events(half).get(2)).get("from_zone"), half);
        assertEquals(2, records.size(), "the cost half and the effect half: " + records);
        assertEquals("resolved", payload(half).get("outcome"), half);
    }

    /** Activate the Poltergeist's "Pay 1 life" at 1 life, then run the check that ends the game. */
    private static void payLifeToZero(Game game, Player payer) {
        payer.setLife(1, null);
        Card poltergeist = TestBoard.inPlay(game, payer, "Mischievous Poltergeist");
        SpellAbility regenerate = TestBoard.ability(poltergeist, ApiType.Regenerate);
        regenerate.setActivatingPlayer(payer);
        assertTrue(ComputerUtil.handlePlayingSpellAbility(payer, regenerate, null),
                "the AI paid the life");
        assertEquals(0, payer.getLife());
        game.getAction().checkStateEffects(false);
        assertTrue(game.isGameOver(), "the payer lost");
    }

    /** Step the engine's main loop until the game is over. */
    private static void playUntilTheGameEnds(Game game) {
        PhaseHandler phases = game.getPhaseHandler();
        for (int step = 0; step < 200 && !game.isGameOver(); step++) {
            phases.mainLoopStep();
        }
        assertTrue(game.isGameOver(), "the defender drew from an empty library");
    }

    private static String only(List<String> records) {
        assertEquals(1, records.size(), records.toString());
        return records.get(0);
    }

    private static boolean isMarker(String line) {
        return line.contains("\"kind\":\"turn_action\"") && line.contains("\"subkind\":\"game_end\"");
    }
}
