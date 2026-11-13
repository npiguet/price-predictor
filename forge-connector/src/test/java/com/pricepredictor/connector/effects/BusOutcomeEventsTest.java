package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.withEventOfType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The outcomes a resolution or combat record reads off the bus besides the
 * moves themselves (C1 #15, V2 #3): a permanent destroyed, a permanent
 * sacrificed, a card turning face up, a player losing the game. Each was
 * written only on a trigger's or a replacement's record, from its run
 * parameters, and never on the record of what did it.
 */
@ExtendWith(ForgeExtension.class)
class BusOutcomeEventsTest {

    private static final String WHY = "for the resolution and state-based-loop hooks";

    @TempDir
    Path tempDir;

    /** Murder destroys Grizzly Bears: {@code destroyed}, then the bears' move. */
    @Test
    void aDestroySpellsHalfSaysThePermanentWasDestroyed() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card murder = TestBoard.inHand(game, caster, "Murder");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, murder, caster, bears);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/m/murder.txt");
        assertEquals(List.of("destroyed", "zone_change"), typesNaming(half, Refs.of(bears)), half);
    }

    /**
     * A Lightning Bolt's lethal damage: the state-based check destroys the
     * bears, and the {@code destroyed} joins the record the death joins,
     * the Bolt's held half.
     */
    @Test
    void aStateBasedDestructionJoinsTheRecordItsDeathJoins() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card bolt = TestBoard.inHand(game, caster, "Lightning Bolt");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, bolt, caster, bears);
            game.getStack().resolveStack();
            for (int step = 0; step < 10 && game.getCardsIn(forge.game.zone.ZoneType.Battlefield)
                    .stream().anyMatch(c -> c.getId() == bears.getId()); step++) {
                game.getPhaseHandler().mainLoopStep();
            }
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        assertEquals(List.of("damage_dealt", "destroyed", "zone_change"),
                typesNaming(half, Refs.of(bears)), half);
    }

    /** Cruel Edict: the opponent sacrifices its creature, {@code sacrificed} then the move. */
    @Test
    void aSacrificeSpellsHalfSaysThePermanentWasSacrificed() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card edict = TestBoard.inHand(game, caster, "Cruel Edict");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, edict, caster, opponent);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/c/cruel_edict.txt");
        assertEquals(List.of("sacrificed", "zone_change"), typesNaming(half, Refs.of(bears)), half);
    }

    /**
     * Break Open turns the opponent's face-down Battering Craghorn face up:
     * a {@code face_change} to {@code front}, in FaceState's spelling.
     */
    @Test
    void aCardTurnedFaceUpIsAFaceChangeToItsFront() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card craghorn = TestBoard.inPlay(game, game.getPlayers().get(1), "Battering Craghorn");
        craghorn.turnFaceDown(true);
        Card breakOpen = TestBoard.inHand(game, caster, "Break Open");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, breakOpen, caster, craghorn);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        assertTrue(!craghorn.isFaceDown(), "Break Open turned the Craghorn face up");
        String half = only(records, "resolution", "cardsfolder/b/break_open.txt");
        List<Map<String, Object>> turned = events(half).stream()
                .filter(event -> "face_change".equals(event.get("type"))).toList();
        assertEquals(1, turned.size(), half);
        assertEquals(List.of(Refs.of(craghorn)), turned.get(0).get("subjects"), half);
        assertEquals("front", params(turned.get(0)).get("to_state"), half);
    }

    /**
     * A Lightning Bolt takes the opponent from 3 life to 0: the game's
     * outcome names the loser, with Forge's reason, before the winner.
     */
    @Test
    void theGamesOutcomeNamesItsLoser() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        opponent.setLife(3, null);
        SpellAbility bolt = TestBoard.spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(opponent);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            TestBoard.castAndResolve(game, bolt);
            // The game's end publishes its outcome on the bus.
            assertTrue(game.isGameOver(), "the opponent lost");
            collector.finishGame(Set.of());
        });

        List<String> outcome = withEventOfType(records, "player_lost");
        assertEquals(1, outcome.size(), records.toString());
        List<Map<String, Object>> told = events(outcome.get(0)).stream()
                .filter(event -> event.get("type").toString().startsWith("player_")).toList();
        assertEquals(List.of("player_lost", "player_won"),
                told.stream().map(event -> event.get("type")).toList(), outcome.get(0));
        assertEquals(List.of(Refs.of(opponent)), told.get(0).get("subjects"), outcome.get(0));
        assertEquals("LifeReachedZero", params(told.get(0)).get("reason"), outcome.get(0));
    }

    /** The types of the record's events naming {@code ref}, in order. */
    private static List<Object> typesNaming(String record, String ref) {
        return events(record).stream()
                .filter(event -> ((List<?>) event.get("subjects")).contains(ref))
                .map(event -> event.get("type")).toList();
    }
}
