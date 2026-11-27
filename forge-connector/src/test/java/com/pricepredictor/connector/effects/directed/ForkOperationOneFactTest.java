package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.ColourTypeOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.card.ColorSet;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code pt}, {@code colour_type}, {@code counters}, {@code tap} and {@code life}
 * operations each change exactly their one fact, and a life edit fires no
 * life-gain trigger.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationOneFactTest {

    @Test
    void ptSetsTheBaseToughnessAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".base_pt"),
                board.edit(new PtOperation(bears, null, 3)));
        assertEquals(3, bears.getNetToughness());
        assertEquals(2, bears.getNetPower());
    }

    @Test
    void colourSetChangesTheColoursAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".colors"),
                board.edit(ColourTypeOperation.colours(bears, ColorSet.fromNames("blue"))));
        assertTrue(bears.getColor().hasBlue());
        assertFalse(bears.getColor().hasGreen());
    }

    @Test
    void countersOnACardChangeItsCountersAndTheBoostTheyGive() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".counters", id(bears) + ".pt_boosts"),
                board.edit(new CountersOperation(bears, CounterEnumType.P1P1, 2)));
        assertEquals(4, bears.getNetPower());
    }

    @Test
    void countersOnAPlayerChangeThatCountAlone() {
        Board board = Board.main();
        assertEquals(Set.of("P1.poison"),
                board.edit(new CountersOperation(board.opponent, CounterEnumType.POISON, 3)));
    }

    @Test
    void tapChangesTheTappedFlagAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".tapped"), board.edit(new TapOperation(bears, true)));
        assertTrue(bears.isTapped());
    }

    @Test
    void lifeChangesTheTotalAloneAndFiresNoLifeGainTrigger() {
        Board board = Board.main();
        board.creature(board.opponent, "Ajani's Pridemate");
        assertEquals(Set.of("P1.life"), board.edit(new LifeOperation(board.opponent, 25)));
        assertEquals(25, board.opponent.getLife());
        assertNoLifeGainCounted(board);
    }

    /** The life edit fired no life-gain trigger and counted no life gained this turn. */
    private static void assertNoLifeGainCounted(Board board) {
        assertFalse(board.triggered(), "Ajani's Pridemate triggered on a silent life edit");
        assertEquals(0, board.opponent.getLifeGainedThisTurn());
    }

    @Test
    void theLifeBoardCanTellAGainThatFires() {
        Board board = Board.main();
        board.creature(board.opponent, "Ajani's Pridemate");
        board.opponent.gainLife(5, null, null);
        assertTrue(board.triggered());
    }
}
