package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code replace_card} operation changes the card alone (T086): the replacement
 * carries the permanent's state, effects, pair, history and designations, and what
 * named the old card names the new one.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationReplaceCardTest {

    @Test
    void replacingABlockerCarriesItsStateAndChangesTheCardAlone() {
        Board board = Board.blocked();
        board.bears.setDamage(1);
        board.bears.setCounters(CounterEnumType.P1P1, 1);
        Map<String, Object> before = BoardStates.of(board.game);

        new ReplaceCardOperation(board.bears, "Centaur Courser").apply(board.game);

        Card courser = board.opponent.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals("Centaur Courser")).findFirst().orElseThrow();
        Map<String, Object> after = BoardStates.of(board.game);
        // The before board with the courser's id wherever the bears' stood:
        // what is left differs only in what makes the card that card.
        Set<String> left = changes(renamed(before, id(board.bears), id(courser)), after);
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines");
        assertTrue(left.contains(id(courser) + ".name"), "the card itself changed: " + left);
        for (String change : left) {
            assertTrue(change.startsWith(id(courser) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        assertNull(board.bears.getZone());
        assertTrue(board.game.getCombat().isBlocking(courser, board.giant));
        assertEquals(courser, board.holy.getEntityAttachedTo());
        assertEquals(1, courser.getDamage());
        assertFalse(board.triggered());
    }

    /**
     * A replacement is the same permanent in every respect but the card: a
     * resolved pump and a keyword granted until end of turn, its soulbond
     * pair and what it attacked this turn carry over; an anthem's rows are
     * the anthem's to apply anew, and a noncreature it no longer applies to
     * keeps none of them.
     */
    @Test
    void aReplacementCarriesThePermanentsEffectsPairAndHistory() {
        Board board = Board.main();
        Card anthem = board.permanent(board.you, "Glorious Anthem");
        Card bears = board.creature(board.you, "Grizzly Bears");
        Card partner = board.creature(board.you, "Hill Giant");
        bears.addPTBoost(3, 3, board.game.getNextTimestamp(), 0);
        bears.addChangedCardKeywords(List.of("Flying"), null, false, board.game.getNextTimestamp(), null);
        bears.setPairedWith(partner);
        partner.setPairedWith(bears);
        bears.getDamageHistory().setCreatureAttackedThisCombat(board.opponent, 0);
        board.game.getAction().checkStaticAbilities();
        assertEquals(2 + 1 + 3, bears.getNetPower());

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card != partner && card.getName().equals("Hill Giant")).findFirst().orElseThrow();

        assertEquals(3 + 1 + 3, giant.getNetPower(), "the pump and the anthem, once each");
        assertTrue(giant.hasKeyword("Flying"));
        assertEquals(partner, giant.getPairedWith());
        assertEquals(giant, partner.getPairedWith());
        assertTrue(giant.getDamageHistory().hasAttackedThisTurn(board.opponent));

        board.edit(new ReplaceCardOperation(giant, "Sol Ring"));
        Card ring = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Sol Ring")).findFirst().orElseThrow();
        long anthemRows = ring.getPTBoostTable().cellSet().stream()
                .filter(cell -> cell.getColumnKey() != 0L).count();
        assertEquals(0, anthemRows, "the anthem's row stayed on a noncreature: " + ring.getPTBoostTable()
                + " (anthem " + anthem + ")");
    }

    /** The Hill Giant that replaced {@code old} on {@code owner}'s battlefield. */
    private static Card replacementOf(Board board, Player owner, Card... others) {
        List<Card> skip = List.of(others);
        return owner.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Hill Giant") && !skip.contains(card))
                .findFirst().orElseThrow();
    }

    /**
     * Giant Growth on a creature, then replace_card: the boost the new card
     * took over ends on it at cleanup, the until-end-of-turn command having
     * been rewritten to name it.
     */
    @Test
    void aPumpTheReplacementTookOverEndsAtCleanup() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        SpellAbility growth = TestBoard.inHand(board.game, board.you, "Giant Growth").getFirstSpellAbility();
        growth.setActivatingPlayer(board.you);
        growth.getTargets().add(bears);
        board.game.getStack().addAndUnfreeze(growth);
        board.game.getStack().resolveStack();
        assertEquals(5, bears.getNetPower());

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);
        assertEquals(6, giant.getNetPower(), "the pump carried");
        board.game.getPhaseHandler().devModeSet(PhaseType.MAIN2, board.you, board.game.getPhaseHandler().getTurn());
        board.game.getPhaseHandler().devAdvanceToPhase(PhaseType.CLEANUP);

        assertEquals(3, giant.getNetPower(), "the pump ended at cleanup");
    }

    /** A delayed trigger and a card remembering the old card remember the new one. */
    @Test
    void whatRemembersTheReplacedCardRemembersTheReplacement() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        Card holder = board.creature(board.opponent, "Gray Ogre");
        holder.addRemembered(bears);
        forge.game.trigger.Trigger delayed = forge.game.trigger.TriggerHandler.parseTrigger(
                "Mode$ Phase | Phase$ End of Turn | TriggerDescription$ test", bears, false);
        delayed.addRemembered(bears);
        board.game.getTriggerHandler().registerDelayedTrigger(delayed);

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);

        assertTrue(holder.isRemembered(giant) && !holder.isRemembered(bears), holder.getRemembered().toString());
        assertSame(giant, delayed.getHostCard());
        assertEquals(List.of(giant), delayed.getTriggerRemembered());
    }

    /** Goad, monstrous and renowned are the permanent's designations: carried. */
    @Test
    void aReplacementCarriesGoadMonstrousAndRenowned() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addGoad(board.game.getNextTimestamp(), board.opponent);
        bears.setMonstrous(true);
        bears.setRenowned(true);

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);

        assertTrue(giant.isGoadedBy(board.opponent));
        assertTrue(giant.isMonstrous());
        assertTrue(giant.isRenowned());
    }
}
