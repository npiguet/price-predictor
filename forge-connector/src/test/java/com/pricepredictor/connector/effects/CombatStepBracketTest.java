package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.card.Card;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A combat record's snapshot is the board as its damage step began (FR-027,
 * story 2's acceptance 6).
 *
 * <p>Played on the real engine, because what is under test is when the
 * snapshot is taken relative to the engine's own steps: the phase event, the
 * damage assignment and the damage dealt, in the order {@code
 * PhaseHandler.mainLoopStep} runs them. A snapshot read at the first damage
 * event already shows that event's damage marked.
 */
@ExtendWith(ForgeExtension.class)
class CombatStepBracketTest {

    @TempDir
    Path tempDir;

    /**
     * White Knight (2/2, first strike) attacks and Hill Giant (3/3) blocks:
     * the first-strike step marks 2 on the giant, the regular step 3 on the
     * knight. Each step's record shows none of its own damage, the first
     * none at all, and both carry the combat's shape and assignment.
     */
    @Test
    void eachDamageStepsSnapshotPrecedesItsDamage() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        game.setAge(GameStage.Play);
        Card knight = TestBoard.inPlay(game, attacking, "White Knight");
        Card giant = TestBoard.inPlay(game, defending, "Hill Giant");
        TestBoard.blockedCombat(game, attacking, defending, knight, giant);

        PhaseHandler phases = game.getPhaseHandler();
        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            for (int step = 0; step < 20 && !phases.is(PhaseType.COMBAT_END); step++) {
                phases.mainLoopStep();
            }
            assertTrue(phases.is(PhaseType.COMBAT_END), "the game reached the end of combat");
        });

        List<Map<String, Object>> combats = records.stream().map(RecordLines::parse)
                .filter(record -> "combat".equals(record.get("kind"))).toList();
        assertEquals(2, combats.size(), records.toString());

        Map<String, Object> firstStrike = combats.get(0);
        assertEquals("first_strike", global(firstStrike).get("combat_substep"));
        assertEquals(0L, damageOn(firstStrike, giant), "no first-strike damage yet");
        assertEquals(0L, damageOn(firstStrike, knight));

        Map<String, Object> regular = combats.get(1);
        assertEquals("regular", global(regular).get("combat_substep"));
        assertEquals(2L, damageOn(regular, giant), "the first-strike step's damage stays marked");
        assertEquals(0L, damageOn(regular, knight), "no regular damage yet");

        for (Map<String, Object> combat : combats) {
            Map<String, Object> payload = asMap(combat.get("payload"));
            assertFalse(((List<?>) payload.get("attackers")).isEmpty(), combat.toString());
            assertFalse(asMap(payload.get("blocks")).isEmpty(), combat.toString());
            assertFalse(asMap(payload.get("assignment_choices")).isEmpty(), combat.toString());
            assertFalse(((List<?>) payload.get("events")).isEmpty(), combat.toString());
        }
    }

    private static Map<String, Object> global(Map<String, Object> record) {
        return asMap(asMap(record.get("state")).get("global"));
    }

    /** The damage marked on {@code card} in the record's snapshot. */
    private static Object damageOn(Map<String, Object> record, Card card) {
        List<?> entities = (List<?>) asMap(record.get("state")).get("entities");
        for (Object entity : entities) {
            Map<String, Object> slot = asMap(entity);
            if (Refs.of(card).equals(slot.get("id"))) {
                return slot.get("damage");
            }
        }
        throw new AssertionError("no " + card + " in " + record);
    }
}
