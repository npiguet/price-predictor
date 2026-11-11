package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.priorityWithTriggers;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A triggered ability's clauses are attributed as a spell's are (X1): Forge
 * executes a trigger's root as a parentless {@code AbilitySub}, which must
 * still read as the root, its sub-abilities as their chain index, and its
 * conditional clauses must reach {@code conditions_met}.
 */
@ExtendWith(ForgeExtension.class)
class TriggerAttributionTest {

    private static final String WHY = "to attribute a trigger's clause events";
    private static final String FAITHFUL = "cardsfolder/f/fallers_faithful.txt";

    @TempDir
    Path tempDir;

    /**
     * Faller's Faithful enters and destroys an undamaged Hill Giant, whose
     * controller then draws two: the Giant's death is the trigger's root,
     * the draws its first sub-ability, and the draw's condition is reported
     * as met.
     */
    @Test
    void anEnteringDestroyTriggersEventsAreAttributedToItsClauses() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.inPlay(game, opponent, "Hill Giant");
        for (int i = 0; i < 4; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, opponent, "Forest"), null, null);
        }
        Card faithful = TestBoard.inHand(game, caster, "Faller's Faithful");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(faithful, null);
            castAndResolve(game, spellOf(onStack, caster));
            priorityWithTriggers(game);
            while (!game.getStack().isEmpty()) {
                game.getStack().resolveStack();
                priorityWithTriggers(game);
            }
            nextStep(game);
        });

        assertTrue(opponent.getCardsIn(ZoneType.Graveyard).stream()
                .anyMatch(card -> card.getId() == giant.getId()), "the trigger destroyed the Giant");
        List<String> triggerHalves = records(all, "resolution").stream()
                .filter(r -> FAITHFUL.equals(RecordLines.scriptOf(r))
                        && "trigger".equals(RecordLines.abilityKeys(r).get(0).get("trait_kind")))
                .toList();
        assertEquals(1, triggerHalves.size(), all.toString());
        String half = triggerHalves.get(0);

        Map<String, Object> death = events(half).stream()
                .filter(e -> "zone_change".equals(e.get("type"))
                        && List.of("E" + giant.getId()).equals(e.get("subjects")))
                .findFirst().orElseThrow(() -> new AssertionError(half));
        assertEquals("graveyard", params(death).get("to_zone"), half);
        assertEquals("root", death.get("attributed_to"), half);

        List<Map<String, Object>> draws = events(half).stream()
                .filter(e -> "card_drawn".equals(e.get("type"))).toList();
        assertEquals(2, draws.size(), half);
        draws.forEach(draw -> assertEquals("0", draw.get("attributed_to"), half));

        assertEquals(List.of(Map.of("segment", "0", "met", true)),
                payload(half).get("conditions_met"), half);
    }
}
