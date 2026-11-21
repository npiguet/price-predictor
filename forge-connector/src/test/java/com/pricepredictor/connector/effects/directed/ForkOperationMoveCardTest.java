package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.entity;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code move_card} operation moves or replaces a card between zones silently
 * (T086a): no draw, discard or descent is counted and no trigger fires.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationMoveCardTest {

    @Test
    void movingFromLibraryToHandDrawsNothingAndKeepsTheLibraryOrder() {
        Board board = Board.main();
        List<Card> library = board.library(board.you, "Forest", "Island", "Mountain");
        Card island = library.get(1);
        Map<String, Object> before = BoardStates.of(board.game);

        Set<String> changed = board.edit(MoveCardOperation.move(island, ZoneType.Hand, null));

        for (String change : changed) {
            assertTrue(Set.of("P0.hand", "P0.library", "+" + id(island)).contains(change)
                            || (change.startsWith("+link ") && change.contains("=" + id(island))),
                    "moving changed " + change + "; all: " + changed);
        }
        assertTrue(changed.containsAll(Set.of("P0.hand", "P0.library", "+" + id(island))));
        assertEquals(List.of(library.get(0), library.get(2)),
                List.copyOf(board.you.getCardsIn(ZoneType.Library)));
        assertNothingDrawnCountedOrTriggered(board, before);
    }

    /** No draw counted, the global members as {@code before}, and no trigger fired. */
    private static void assertNothingDrawnCountedOrTriggered(Board board, Map<String, Object> before) {
        assertEquals(0, board.you.getNumDrawnThisTurn());
        assertEquals(before.get("global"), BoardStates.of(board.game).get("global"));
        assertFalse(board.triggered());
    }

    @Test
    void movingFromHandToGraveyardFiresNoDiscardTriggerAndNoDescent() {
        Board board = Board.main();
        board.permanent(board.opponent, "Megrim");
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        board.settled();

        Set<String> changed = board.edit(MoveCardOperation.move(bears, ZoneType.Graveyard, null));

        for (String change : changed) {
            assertTrue(Set.of("P0.hand", "P0.graveyard").contains(change)
                            || change.startsWith(id(bears) + "."),
                    "moving changed " + change + "; all: " + changed);
        }
        assertEquals(Set.of(id(bears) + ".zone"), only(changed, id(bears) + "."));
        assertEquals(0, board.you.getDescended());
        assertFalse(board.triggered(), "Megrim triggered on a silent move to the graveyard");
    }

    @Test
    void theDiscardBoardCanTellADiscardThatFires() {
        Board board = Board.main();
        board.permanent(board.opponent, "Megrim");
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        board.settled();
        board.you.discard(bears, null, false, AbilityKey.newMap());
        assertTrue(board.triggered());
    }

    @Test
    void replacingTheTopLibraryCardPutsTheNewCardOnTop() {
        Board board = Board.main();
        List<Card> library = board.library(board.you, "Forest", "Island");
        Card forest = library.get(0);
        Map<String, Object> before = BoardStates.of(board.game);

        Set<String> changed = board.edit(MoveCardOperation.replace(forest, "Lightning Bolt"));

        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        assertEquals("Lightning Bolt", top.getName());
        assertEquals(List.of(top, library.get(1)), List.copyOf(board.you.getCardsIn(ZoneType.Library)));
        assertNull(forest.getZone());
        Map<String, Object> entity = entity(BoardStates.of(board.game), id(top));
        assertEquals("library", entity.get("zone"));
        assertEquals(0L, entity.get("order_rank"));
        for (String change : changed) {
            assertTrue(change.equals("-" + id(forest)) || change.equals("+" + id(top))
                            || change.startsWith("-link ") && change.contains("=" + id(forest))
                            || change.startsWith("+link ") && change.contains("=" + id(top)),
                    "replacing changed " + change + "; all: " + changed);
        }
        assertEquals(entity(before, id(forest)).get("order_rank"), entity.get("order_rank"));
        assertFalse(board.triggered());
    }

    @Test
    void aPlaceBeyondTheZoneFails() {
        Board board = Board.main();
        Card island = board.library(board.you, "Island").get(0);
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        assertThrows(IllegalStateException.class,
                () -> MoveCardOperation.move(bears, ZoneType.Library, 3).apply(board.game));
        assertEquals(List.of(island), List.copyOf(board.you.getCardsIn(ZoneType.Library)));
    }

    /** The changes of one entity or player, by their {@code prefix}. */
    private static Set<String> only(Set<String> changed, String prefix) {
        Set<String> kept = new TreeSet<>();
        for (String change : changed) {
            if (change.startsWith(prefix)) {
                kept.add(change);
            }
        }
        return kept;
    }
}
