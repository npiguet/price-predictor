package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.directed.BoardChanges.entity;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.indexed;
import static com.pricepredictor.connector.effects.directed.BoardChanges.links;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code remove_permanent} operation takes the permanent alone and silently:
 * no dies or leaves trigger fires; its twin removal the way the game
 * does fires.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationRemovePermanentTest {

    @Test
    void removingABlockerWithAnAuraTakesItAloneAndFiresNothing() {
        Board board = Board.blocked();
        Map<String, Object> before = BoardStates.of(board.game);
        String bears = id(board.bears);

        Set<String> changed = board.edit(new RemovePermanentOperation(board.bears));

        // The Aura's static affects nothing now (its line's affected list),
        // and each later permanent moves up one place in timestamp order.
        Set<String> expected = new TreeSet<>(Set.of(
                "-" + bears, id(board.holy) + ".attached_to", id(board.holy) + ".lines",
                id(board.giant) + ".combat"));
        Object rank = entity(before, bears).get("order_rank");
        addLaterOrderRanks(expected, before, rank);
        addLinksNaming(expected, before, bears);
        assertEquals(expected, new TreeSet<>(changed));
        Map<String, Object> holy = entity(BoardStates.of(board.game), id(board.holy));
        assertNull(holy.get("attached_to"));
        assertTheBlockIsGone(board);
        assertNull(board.bears.getZone());
        assertFalse(board.triggered(), "a dies or leaves trigger fired on a silent removal");
    }

    /** Each permanent placed after {@code rank} in timestamp order moves up one place. */
    private static void addLaterOrderRanks(Set<String> expected, Map<String, Object> before, Object rank) {
        for (Map.Entry<String, Map<String, Object>> other : indexed(before, "entities").entrySet()) {
            if ((Long) other.getValue().get("order_rank") > (Long) rank) {
                expected.add(other.getKey() + ".order_rank");
            }
        }
    }

    /** Each link naming the removed {@code bears} goes. */
    private static void addLinksNaming(Set<String> expected, Map<String, Object> before, String bears) {
        for (String link : links(before)) {
            if (link.contains("=" + bears + ",") || link.contains("=" + bears + "}")) {
                expected.add("-link " + link);
            }
        }
    }

    /** The removed bears block nothing, and the giant has no blocker. */
    private static void assertTheBlockIsGone(Board board) {
        assertFalse(board.game.getCombat().isBlocking(board.bears));
        assertTrue(board.game.getCombat().getBlockers(board.giant).isEmpty());
    }

    @Test
    void theRemovalBoardCanTellADeathThatFires() {
        Board board = Board.blocked();
        board.game.getAction().moveToGraveyard(board.bears, null, null);
        assertTrue(board.triggered());
    }
}
