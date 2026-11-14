package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A whole shard, end to end: the engine plays a spell and a combat with the
 * bracket on the bus and every hook installed, and every line it writes is
 * compared with the expected file, the envelope timestamps alone replaced.
 */
@CharacterizationTest
class ShardCharacterizationTest {

    @TempDir
    Path tempDir;

    /**
     * Lightning Bolt kills a blocker-to-be, then a trampler attacks and is
     * blocked: a cost half, its effect half carrying the death, and the
     * damage step's combat record.
     */
    @Test
    void aSpellAndACombat() throws Throwable {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "the expected shard was captured against the patched ../forge");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card dreadmaw = board.battlefield("Colossal Dreadmaw", you);
        dreadmaw.setSickness(false);
        Card bears = board.battlefield("Grizzly Bears", opponent);
        Card elves = board.battlefield("Llanowar Elves", opponent);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, board.game, "", () -> {
                    board.cast(board.card("Lightning Bolt", you), you, elves);
                    board.game.getStack().resolveStack();
                    board.game.getAction().checkStateEffects(false);

                    board.game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DAMAGE, you);
                    Combat combat = TestBoard.blockedCombat(
                            board.game, you, opponent, dreadmaw, bears);
                    assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
                    combat.dealAssignedDamage();
                    board.game.getAction().checkStateEffects(false);

                    board.game.getPhaseHandler().devModeSet(PhaseType.COMBAT_END, you);
                });

        Characterization.assertMatches(
                "shard-spell-and-combat.jsonl", Characterization.recordLines(records));
    }

    /**
     * Lightning Bolt to an opponent at 3 life ends the game: the record
     * holding the outcome, then the game-end marker, last.
     */
    @Test
    void aGameToItsEnd() throws Throwable {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "the expected shard was captured against the patched ../forge");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        opponent.setLife(3, null);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, board.game, "", () -> {
                    TestBoard.cast(board.game, board.card("Lightning Bolt", you), you, opponent);
                    board.game.getStack().resolveStack();
                    board.game.getAction().checkStateEffects(false);
                    // What Match.startGame publishes once the game is over.
                    board.game.fireEvent(new forge.game.event.GameEventGameFinished());
                });

        assertTrue(board.game.isGameOver(), "the bolt ended the game");
        Characterization.assertMatches(
                "shard-game-end.jsonl", Characterization.recordLines(records));
    }
}
