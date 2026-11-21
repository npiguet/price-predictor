package com.pricepredictor.connector.effects.directed;

import forge.ai.simulation.GameCopier;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.entity;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code add_permanent} operation adds the permanent as asked and silently
 * (T086a), and on a real copy the first state-based check sees an added 0/0 die.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationAddPermanentTest {

    @Test
    void addingAnElvishVisionaryDrawsNothingAndArrivesAsAsked() {
        Board board = Board.main();
        board.library(board.you, "Forest", "Island");
        Map<String, Object> before = BoardStates.of(board.game);

        new AddPermanentOperation("Elvish Visionary", board.you, board.you, true, true,
                Map.of(CounterEnumType.P1P1, 1)).apply(board.game);

        Card visionary = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals("Elvish Visionary")).findFirst().orElseThrow();
        Map<String, Object> after = BoardStates.of(board.game);
        String added = id(visionary);
        for (String change : changes(before, after)) {
            assertTrue(change.equals("+" + added)
                            || (change.startsWith("+link ") && change.contains("=" + added)),
                    "adding changed " + change);
        }
        Map<String, Object> entity = entity(after, added);
        assertArrivedTappedSickWithOneCounter(entity);
        assertEquals(0, board.you.getCardsIn(ZoneType.Hand).size());
        assertFalse(board.triggered(), "Elvish Visionary's enters trigger fired on a silent add");
    }

    /** The added permanent's snapshot: tapped, summoning-sick, one +1/+1 counter. */
    private static void assertArrivedTappedSickWithOneCounter(Map<String, Object> entity) {
        assertEquals(true, entity.get("tapped"));
        assertEquals(true, entity.get("sick"));
        assertEquals(List.of(1L), new ArrayList<>(((Map<?, ?>) entity.get("counters")).values()));
    }

    @Test
    void addingUnderAnotherControllerPutsItOnTheirBattlefield() {
        Board board = Board.main();
        new AddPermanentOperation("Grizzly Bears", board.you, board.opponent, false, false,
                Map.of()).apply(board.game);
        Card bears = board.opponent.getCardsIn(ZoneType.Battlefield).get(0);
        assertYoursUnderTheOpponent(board, bears);
        assertFalse(bears.isSick());
        assertEquals(Boolean.FALSE, ((Map<?, ?>) entity(BoardStates.of(board.game), id(bears)).get("history"))
                        .get("entered_this_turn"),
                "a permanent that is not summoning-sick has been here since an earlier turn");
    }

    /** You own {@code bears}; the opponent controls it. */
    private static void assertYoursUnderTheOpponent(Board board, Card bears) {
        assertEquals(board.you, bears.getOwner());
        assertEquals(board.opponent, bears.getController());
    }

    /**
     * A permanent the edit adds and the first state-based check takes away
     * (a 0/0 Walking Ballista) is seen dying: Blood Artist triggers. The
     * copy's last-known battlefield was taken before the edit, so without
     * the edit taking it anew the death looked back at nothing.
     */
    @Test
    void anAddedPermanentTheFirstCheckKillsIsSeenDying() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        TestBoard.inPlay(live, you, "Blood Artist");
        live.getTriggerHandler().resetActiveTriggers(false, null);
        live.copyLastState();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();

        AddPermanentOperation.of("Walking Ballista", copier.find(you), false).apply(fork);
        fork.getAction().checkStateEffects(true);
        fork.getTriggerHandler().runWaitingTriggers();

        assertTrue(copier.find(you).getCardsIn(ZoneType.Battlefield).stream()
                .noneMatch(card -> card.getName().equals("Walking Ballista")), "the 0/0 did not die");
        assertTrue(fork.getStack().hasSimultaneousStackEntries() || !fork.getStack().isEmpty(),
                "Blood Artist did not see the Ballista die");
    }

    @Test
    void theAddBoardCanTellAnEnteringThatFires() {
        Board board = Board.main();
        board.library(board.you, "Forest", "Island");
        Card visionary = TestBoard.inHand(board.game, board.you, "Elvish Visionary");
        board.settled();
        board.game.getAction().moveToPlay(visionary, board.you, null, null);
        assertTrue(board.triggered());
    }
}
