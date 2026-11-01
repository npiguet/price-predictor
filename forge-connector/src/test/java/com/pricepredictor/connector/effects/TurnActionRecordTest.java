package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.event.GameEventPlayerPriority;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A turn-based action gets a {@code turn_action} record of its own (FR-036, C7).
 *
 * <p>Played on the real engine: {@code PhaseHandler.mainLoopStep} carries a
 * game from one player's end step through the other's draw, so the phase
 * events, the turn-based actions and the priorities arrive in the order Forge
 * publishes them, which is what the window is built on.
 */
@ExtendWith(ForgeExtension.class)
class TurnActionRecordTest {

    @TempDir
    Path tempDir;

    /** The turn-action records among {@code records}, in the order written. */
    private static List<Map<String, Object>> turnActions(List<String> records) {
        return records.stream().map(RecordLines::parse)
                .filter(record -> "turn_action".equals(record.get("kind")))
                .toList();
    }

    private static List<String> subkinds(List<Map<String, Object>> records) {
        return records.stream().map(record -> (String) record.get("subkind")).toList();
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> eventsOf(Map<String, Object> record) {
        return (List<Map<String, Object>>) RecordLines.asMap(record.get("payload")).get("events");
    }

    private static List<String> typesOf(Map<String, Object> record) {
        return eventsOf(record).stream().map(event -> (String) event.get("type")).toList();
    }

    private static Map<String, Object> only(List<Map<String, Object>> records, String subkind) {
        List<Map<String, Object>> matching = records.stream()
                .filter(record -> subkind.equals(record.get("subkind"))).toList();
        assertEquals(1, matching.size(), subkind + " records: " + records);
        return matching.get(0);
    }

    private static void giveCards(Game game, Player owner, ZoneType zone, String name, int count) {
        for (int i = 0; i < count; i++) {
            game.getAction().moveTo(zone, TestBoard.card(game, owner, name), null, null);
        }
    }

    /**
     * One player's cleanup into the other's precombat main phase: every one
     * of the six turn-based actions, each in a record of its own, with the
     * active player as the actor, no acting line, and the events the action
     * is made of.
     *
     * <p>The ending player holds eight cards and discards one; the next one's
     * Breezekeeper phases out, the day becomes night because the ending
     * player cast nothing, a tapped Forest untaps, a card is drawn, and
     * History of Benalia gets a lore counter.
     */
    @Test
    void aTurnWrapWritesEachTurnBasedActionItsOwnRecord() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.END_OF_TURN);
        Player next = game.getPlayers().get(0);
        Player ending = game.getPlayers().get(1);
        // Turn 1 is the ending player's, so the next one's draw is not skipped.
        game.getPhaseHandler().devModeSet(PhaseType.END_OF_TURN, ending, 1);
        game.setAge(GameStage.Play);

        giveCards(game, ending, ZoneType.Hand, "Craw Wurm", 8);
        giveCards(game, ending, ZoneType.Library, "Craw Wurm", 2);
        giveCards(game, next, ZoneType.Library, "Craw Wurm", 2);
        Card forest = TestBoard.inPlay(game, next, "Forest");
        forest.setTapped(true);
        Card breezekeeper = TestBoard.inPlay(game, next, "Breezekeeper");
        Card saga = TestBoard.inPlay(game, next, "History of Benalia");
        game.getTriggerHandler().clearWaitingTriggers();
        game.setDayTime(false);

        PhaseHandler phases = game.getPhaseHandler();
        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            for (int step = 0; step < 100
                    && !(phases.is(PhaseType.MAIN1) && phases.getPlayerTurn() == next); step++) {
                phases.mainLoopStep();
            }
            assertTrue(phases.is(PhaseType.MAIN1) && phases.getPlayerTurn() == next,
                    "the game reached the next player's main phase");
            // Its priority, which the main loop would give next.
            game.fireEvent(new GameEventPlayerPriority(
                    PlayerView.get(next), PhaseType.MAIN1, PlayerView.get(next)));
        });

        List<Map<String, Object>> actions = turnActions(records);
        assertEquals(
                List.of("cleanup_discard", "phasing", "day_night", "untap", "draw", "lore_counter"),
                subkinds(actions), actions.toString());
        for (Map<String, Object> action : actions) {
            assertNull(action.get("ability"), action.toString());
            assertNull(action.get("moment"), action.toString());
            assertFalse(eventsOf(action).isEmpty(), action.toString());
        }

        Map<String, Object> discard = only(actions, "cleanup_discard");
        assertEquals(Refs.of(ending), discard.get("actor_player"));
        assertEquals(List.of("zone_change", "card_discarded"), typesOf(discard), discard.toString());
        assertEquals("hand", RecordLines.params(eventsOf(discard).get(0)).get("from_zone"));

        Map<String, Object> phasing = only(actions, "phasing");
        assertEquals(Refs.of(next), phasing.get("actor_player"));
        Map<String, Object> phasedOut = eventsOf(phasing).get(0);
        assertEquals(List.of(Refs.of(breezekeeper)), phasedOut.get("subjects"));
        assertEquals(true, RecordLines.params(phasedOut).get("out"));

        Map<String, Object> dayNight = only(actions, "day_night");
        assertEquals(List.of("day_night_changed"), typesOf(dayNight));
        assertEquals("night", RecordLines.params(eventsOf(dayNight).get(0)).get("to"));

        Map<String, Object> untap = only(actions, "untap");
        assertTrue(eventsOf(untap).stream().anyMatch(event -> "untapped".equals(event.get("type"))
                        && List.of(Refs.of(forest)).equals(event.get("subjects"))),
                untap.toString());
        assertTrue(typesOf(untap).stream().allMatch("untapped"::equals), untap.toString());

        // Acceptance 10: the draw step's record holds the draw.
        Map<String, Object> draw = only(actions, "draw");
        assertEquals(Refs.of(next), draw.get("actor_player"));
        assertEquals(List.of("zone_change", "card_drawn"), typesOf(draw), draw.toString());
        assertEquals(List.of(Refs.of(next)), eventsOf(draw).get(1).get("subjects"));

        Map<String, Object> lore = only(actions, "lore_counter");
        Map<String, Object> counter = eventsOf(lore).get(0);
        assertEquals(List.of(Refs.of(saga)), counter.get("subjects"));
        assertEquals("LORE", RecordLines.params(counter).get("counter_type"));
        assertEquals(1L, RecordLines.params(counter).get("delta"));
    }

    /**
     * The window closes at the priority: a card drawn afterwards, in the same
     * step, is an ability's draw or nobody's, never the turn's.
     */
    @Test
    void theWindowClosesAtTheNextPriority() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        giveCards(game, active, ZoneType.Library, "Craw Wurm", 3);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            TestBoard.enterStep(game, PhaseType.DRAW);
            active.drawCard();
            game.fireEvent(new GameEventPlayerPriority(
                    PlayerView.get(active), PhaseType.DRAW, PlayerView.get(active)));
            active.drawCard();
            TestBoard.nextStep(game);
        });

        List<Map<String, Object>> actions = turnActions(records);
        assertEquals(List.of("draw"), subkinds(actions), records.toString());
        assertEquals(1, typesOf(actions.get(0)).stream().filter("card_drawn"::equals).count(),
                actions.toString());
    }

    /**
     * A cleanup whose hand is within its maximum, or a main phase with no
     * Saga, has no action to record and writes nothing.
     */
    @Test
    void aStepWhoseActionCannotHappenWritesNothing() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.END_OF_TURN);
        Player active = game.getPlayers().get(0);
        giveCards(game, active, ZoneType.Hand, "Craw Wurm", 7);

        // Nothing at all is written, so there is no shard to read.
        assertThrows(NoSuchFileException.class, () -> ShardRun.observed(tempDir, game, collector -> {
            TestBoard.enterStep(game, PhaseType.MAIN1);
            TestBoard.enterStep(game, PhaseType.CLEANUP);
            // A discard by hand, in a window that never opened.
            active.discard(active.getCardsIn(ZoneType.Hand).get(0), null, false,
                    AbilityKey.newMap());
            TestBoard.enterStep(game, PhaseType.END_OF_TURN);
        }));
    }

    /**
     * What a step publishes besides its actions is not taken: a power a
     * static recomputed after a phasing, a non-lore counter, a card leaving
     * the hand during a draw step.
     */
    @Test
    void onlyTheEventsAnActionIsMadeOfAreTaken() {
        assertEquals(TurnActionSubkind.PHASING, TurnActionWindow.actionOf(PhaseType.UNTAP,
                new EffectEvent(EventType.PHASED).subject("E1").param("out", true)));
        assertNull(TurnActionWindow.actionOf(PhaseType.UNTAP,
                new EffectEvent(EventType.PT_CHANGE).subject("E1")));
        assertNull(TurnActionWindow.actionOf(PhaseType.UNTAP,
                new EffectEvent(EventType.TAPPED).subject("E1")));
        assertNull(TurnActionWindow.actionOf(PhaseType.MAIN1,
                new EffectEvent(EventType.COUNTER_CHANGE).subject("E1")
                        .param("counter_type", "P1P1").param("delta", 1)));
        assertNull(TurnActionWindow.actionOf(PhaseType.DRAW,
                new EffectEvent(EventType.ZONE_CHANGE).subject("E1")
                        .param("from_zone", "hand").param("to_zone", "graveyard")));
        assertEquals(TurnActionSubkind.DRAW, TurnActionWindow.actionOf(PhaseType.DRAW,
                new EffectEvent(EventType.ZONE_CHANGE).subject("E1")
                        .param("from_zone", "graveyard").param("to_zone", "hand")));
        assertEquals(TurnActionSubkind.CLEANUP_DISCARD, TurnActionWindow.actionOf(
                PhaseType.CLEANUP, new EffectEvent(EventType.ZONE_CHANGE).subject("E1")
                        .param("from_zone", "hand").param("to_zone", "exile")));
        assertNull(TurnActionWindow.actionOf(PhaseType.UPKEEP,
                new EffectEvent(EventType.UNTAPPED).subject("E1")));
    }
}
