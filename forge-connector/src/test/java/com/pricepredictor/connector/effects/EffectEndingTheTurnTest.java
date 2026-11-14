package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An effect that ends the turn or the combat phase, played on the real loop.
 *
 * <p>Time Stop and Mandate of Peace exile every spell on the stack, theirs
 * included, run a state-based check while they resolve, and move the game to
 * another step — a phase event published in the middle of their own
 * resolution, ahead of the resolution's own end. Their effect half is still
 * written, with what they did.
 */
@ExtendWith(ForgeExtension.class)
class EffectEndingTheTurnTest {

    private static final String TIME_STOP = "cardsfolder/t/time_stop.txt";
    private static final String MANDATE = "cardsfolder/m/mandate_of_peace.txt";

    @TempDir
    Path tempDir;

    /**
     * Time Stop resolves: its effect half holds the turn's end and its own
     * card's exile off the stack, and reads {@code resolved} — not the
     * missing half of an item countered. The state-based check it runs while
     * it resolves is its own: a creature left with lethal damage dies there,
     * into its half.
     */
    @Test
    void endingTheTurnKeepsTheSpellsEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card timeStop = TestBoard.inHand(game, caster, "Time Stop");
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the resolution and state-based-loop hooks", () -> {
                    castWithoutTarget(game, timeStop, caster);
                    // Lethal damage no check has looked at yet: the one Time
                    // Stop runs while it resolves is the first.
                    bears.setDamage(2);
                    game.getStack().resolveStack();
                    assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                            .anyMatch(card -> card.getId() == bears.getId()), "the bears died");
                    // The next turn's first step closes the held half.
                    stepUntilPhase(game, PhaseType.UPKEEP);
                });

        String half = only(records, "resolution", TIME_STOP);
        assertEquals("resolved", outcomeOf(half), half);
        List<String> types = eventTypes(half);
        assertTrue(types.contains("turn_ended"), "the turn's end is the spell's: " + half);
        assertTrue(events(half).stream().anyMatch(event -> "zone_change".equals(event.get("type"))
                        && "stack".equals(params(event).get("from_zone"))
                        && "exile".equals(params(event).get("to_zone"))),
                "the spell's own exile off the stack: " + half);
        assertTrue(half.contains("{\"type\":\"zone_change\",\"subjects\":[\"E" + bears.getId()
                        + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""),
                "the death in the check it ran: " + half);
        assertEquals(1, RecordLines.records(records, "resolution").stream()
                .filter(record -> TIME_STOP.equals(RecordLines.scriptOf(record))).count(), records.toString());
    }

    /**
     * The cleanup step Time Stop skips to is the step's own, not the spell's:
     * its controller holds nine cards once it is cast, and a Giant Growth'd
     * Grizzly Bears.
     * The discard to hand size is a {@code cleanup_discard} turn-action
     * record, as in any cleanup, and the pump wearing off reaches no record,
     * as in any cleanup; Time Stop's half keeps the turn's end and its exile.
     */
    @Test
    void theCleanupStepTimeStopSkipsToIsTheSteps() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        TestBoard.cast(game, TestBoard.inHand(game, caster, "Giant Growth"), caster, bears);
        game.getStack().resolveStack();
        game.getAction().checkStateEffects(false);
        assertEquals(5, bears.getNetPower(), "the bears are pumped until end of turn");
        Card timeStop = TestBoard.inHand(game, caster, "Time Stop");
        for (int card = 0; card < 9; card++) {
            TestBoard.inHand(game, caster, "Mountain");
        }
        assertEquals(10, caster.getCardsIn(ZoneType.Hand).size(), "nine once Time Stop is cast");

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the resolution and state-based-loop hooks", () -> {
                    castWithoutTarget(game, timeStop, caster);
                    game.getStack().resolveStack();
                    stepUntilPhase(game, PhaseType.UPKEEP);
                });

        assertEquals(7, caster.getCardsIn(ZoneType.Hand).size(), "the cleanup discarded two");
        String half = only(records, "resolution", TIME_STOP);
        List<String> types = eventTypes(half);
        assertTrue(types.contains("turn_ended"), half);
        assertTrue(!types.contains("card_discarded") && !types.contains("pt_change"),
                "the cleanup's discard and the pump wearing off are not the spell's: " + half);
        assertTrue(events(half).stream().noneMatch(event -> "hand".equals(params(event).get("from_zone"))),
                half);
        List<String> discards = records.stream()
                .filter(record -> "cleanup_discard".equals(RecordLines.parse(record).get("subkind")))
                .toList();
        assertEquals(1, discards.size(), records.toString());
        assertEquals(2, eventTypes(discards.get(0)).stream()
                .filter("card_discarded"::equals).count(), discards.get(0));
        assertTrue(events(discards.get(0)).stream()
                        .allMatch(event -> "unresolved".equals(event.get("attributed_to"))),
                "a turn-based action's events name no clause: " + discards.get(0));
    }

    /**
     * Mandate of Peace ends the combat phase: its own state-based check and
     * the phase end's both run inside its resolution, and its effect half
     * keeps what it did, written once.
     */
    @Test
    void endingTheCombatPhaseKeepsTheSpellsEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card mandate = TestBoard.inHand(game, defending, "Mandate of Peace");
        forge.game.combat.Combat combat = new forge.game.combat.Combat(attacking);
        combat.addAttacker(bears, defending);
        game.getPhaseHandler().setCombat(combat);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the resolution and state-based-loop hooks", () -> {
                    castWithoutTarget(game, mandate, defending);
                    stepUntilOffTheStack(game);
                    stepUntilPhase(game, PhaseType.END_OF_TURN);
                });

        String half = only(records, "resolution", MANDATE);
        assertEquals("resolved", outcomeOf(half), half);
        assertTrue(events(half).stream().anyMatch(event -> "zone_change".equals(event.get("type"))
                        && "stack".equals(params(event).get("from_zone"))
                        && "exile".equals(params(event).get("to_zone"))),
                "the spell's own exile off the stack: " + half);
        // The combat it ended, named by its attacking player, on the real path.
        List<Map<String, Object>> ended = events(half).stream()
                .filter(event -> "combat_ended".equals(event.get("type"))).toList();
        assertEquals(1, ended.size(), half);
        assertEquals(List.of(Refs.of(attacking)), ended.get(0).get("subjects"), half);
    }

    /** A card's spell put on the stack the way a cast puts it there, with no target. */
    static SpellAbility castWithoutTarget(Game game, Card card, Player caster) {
        Card onStack = game.getAction().moveToStack(card, null);
        SpellAbility spell = TestBoard.spellOf(onStack, caster);
        game.getStack().add(spell);
        return spell;
    }

    /** Play the real loop until the stack is empty and nothing resolves. */
    private static void stepUntilOffTheStack(Game game) {
        for (int step = 0; step < 40 && !game.getStack().isEmpty(); step++) {
            game.getPhaseHandler().mainLoopStep();
        }
        assertTrue(game.getStack().isEmpty(), "the stack resolved: " + game.getStack());
        assertTrue(game.getCardsIn(ZoneType.Stack).isEmpty(), "nothing is left in the stack zone");
    }

    /** Play the real loop until {@code phase} has begun. */
    private static void stepUntilPhase(Game game, PhaseType phase) {
        for (int step = 0; step < 60 && game.getPhaseHandler().getPhase() != phase; step++) {
            game.getPhaseHandler().mainLoopStep();
        }
        assertEquals(phase, game.getPhaseHandler().getPhase());
    }
}
