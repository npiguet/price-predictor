package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.ForkRefusal;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the {@code replace_card} operation refuses rather than carry over
 * unfaithfully, the board left unchanged, and the replacement it lets through.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationReplaceCardRefusalTest {

    private static void assertRefused(Board board, Card old, String reason) {
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(old, "Hill Giant").apply(board.game));
        assertEquals(reason, refusal.reason());
        assertTrue(old.getZone() != null && old.getZone().is(ZoneType.Battlefield), "the refusal changed the board");
    }

    @Test
    void aFaceDownPermanentIsRefused() {
        Board board = Board.main();
        Card bears = TestBoard.card(board.game, board.you, "Grizzly Bears");
        bears.turnFaceDownNoUpdate();
        board.you.getZone(ZoneType.Battlefield).add(bears);
        assertRefused(board, bears, ReplaceCardOperation.FACE_DOWN);
    }

    @Test
    void aClassAboveItsFirstLevelIsRefused() {
        Board board = Board.main();
        Card ranger = board.permanent(board.you, "Ranger Class");
        ranger.setClassLevel(2);
        assertRefused(board, ranger, ReplaceCardOperation.CLASS_LEVEL);
    }

    @Test
    void aPermanentHoldingAChosenValueIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setChosenType("Elf");
        assertRefused(board, bears, ReplaceCardOperation.CHOSEN_VALUES);
    }

    @Test
    void aPermanentRememberingACardIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addRemembered(board.creature(board.opponent, "Gray Ogre"));
        assertRefused(board, bears, ReplaceCardOperation.LINKED_CARDS);
    }

    @Test
    void aPermanentPhasedOutIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setPhasedOut(board.you);
        assertRefused(board, bears, ReplaceCardOperation.PHASED_OUT);
    }

    /** A pending command that cannot be read (it holds an object it cannot write) leaves the edit refused. */
    @Test
    void aCommandThatCannotBeRewrittenRefusesTheEdit() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        Object unwritable = new Object();
        board.game.getEndOfTurn().addUntil(() -> unwritable.hashCode());
        assertRefused(board, bears, ReplaceCardOperation.UNREWRITABLE_COMMAND);
    }

    /**
     * C4 #1: the damage the replaced creature carries is lethal to a smaller
     * card, which the fork's first check would put into a graveyard; the edit
     * refuses rather than fork that board. Hill Giant with 2 damage becomes
     * Grizzly Bears with 2 damage.
     */
    @Test
    void aReplacementTheCarriedDamageKillsIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.setDamage(2);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    @Test
    void aReplacementTheCarriedCountersKillIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.setCounters(CounterEnumType.M1M1, 2);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    /** A resolved -2/-2 until end of turn leaves Hill Giant a 1/1 and would leave Grizzly Bears a 0/0. */
    @Test
    void aReplacementTheCarriedBoostKillsIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.addPTBoost(-2, -2, board.game.getNextTimestamp(), 0);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        assertFalse(ReplaceCardOperation.replacementDies(giant, "Craw Wurm"), "a 6/4 lives at 2 toughness");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    @Test
    void aReplacementThatSurvivesTheCarriedDamageGoesThrough() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setDamage(1);
        assertFalse(ReplaceCardOperation.replacementDies(bears, "Hill Giant"), "the preconditions' reading");
        new ReplaceCardOperation(bears, "Hill Giant").apply(board.game);
        assertTrue(board.you.getCardsIn(ZoneType.Battlefield).stream()
                .anyMatch(card -> card.getName().equals("Hill Giant") && card.getDamage() == 1));
    }
}
