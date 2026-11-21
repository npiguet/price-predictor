package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;

/**
 * Each fork operation changes exactly its one fact (T086), and the three
 * board-membership operations change it silently (T086a, story 3 acceptance
 * 6): no trigger fires from the edit, and no per-turn count moves.
 *
 * <p>Every check compares the board-only snapshot a record would carry,
 * before and after the edit, member by member: {@link BoardChanges#changes} names each
 * entity, player or global member that differs, each entity that appeared or
 * went, and each link that appeared or went. The edits run on a fresh
 * {@code TestBoard} game, which is what a fork is to its edit.
 *
 * <p>A trigger is caught where Forge first records one: run, it lands among
 * the stack's simultaneous entries. Each silence test has a twin making the
 * same change the way the game does, which fires: the board can tell.
 */
final class ForkOperationSupport {

    private ForkOperationSupport() {
    }

    /**
     * A started two-seat game: you (P0) and the opponent (P1), with whatever
     * each test puts on it, settled so no trigger is waiting before the edit.
     */
    static final class Board {
        final Game game;
        final Player you;
        final Player opponent;
        Card giant;
        Card bears;
        Card holy;

        private Board(PhaseType phase) {
            game = TestBoard.startedTwoPlayerGame(phase);
            you = game.getPlayers().get(0);
            opponent = game.getPlayers().get(1);
        }

        static Board main() {
            return new Board(PhaseType.MAIN1);
        }

        /**
         * Your Hill Giant attacks; the opponent's Grizzly Bears, enchanted by
         * the opponent's Holy Strength, blocks it; a Doomed Dissenter and a
         * Blood Artist watch.
         */
        static Board blocked() {
            Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
            board.giant = board.creature(board.you, "Hill Giant");
            board.bears = board.creature(board.opponent, "Grizzly Bears");
            board.holy = board.permanent(board.opponent, "Holy Strength");
            board.holy.attachToEntity(board.bears, null);
            board.creature(board.opponent, "Doomed Dissenter");
            board.creature(board.opponent, "Blood Artist");
            TestBoard.blockedCombat(new TestBoard.CombatSides(board.game, board.you, board.opponent), board.giant, board.bears);
            board.settled();
            return board;
        }

        /**
         * A permanent since the turn began, with a timestamp of its own as
         * one entering in play has (a test board's are all unset).
         */
        Card creature(Player owner, String name) {
            Card card = TestBoard.inPlay(game, owner, name);
            card.setGameTimestamp(game.getNextTimestamp());
            settled();
            return card;
        }

        Card permanent(Player owner, String name) {
            return creature(owner, name);
        }

        /** Cards put on the bottom in turn, so the list reads top first. */
        List<Card> library(Player owner, String... names) {
            List<Card> cards = new ArrayList<>();
            for (String name : names) {
                cards.add(game.getAction().moveToLibrary(
                        TestBoard.card(game, owner, name), -1, null, null));
            }
            settled();
            return cards;
        }

        /**
         * Nothing waiting, nothing on the stack, and the last-state battlefield
         * current, so a dies trigger can look back at it.
         */
        void settled() {
            game.getTriggerHandler().runWaitingTriggers();
            game.getStack().clearSimultaneousStack();
            game.getTriggerHandler().resetActiveTriggers(false, null);
            game.copyLastState();
        }

        /** Whether a trigger has fired since the board was settled. */
        boolean triggered() {
            game.getTriggerHandler().runWaitingTriggers();
            return game.getStack().hasSimultaneousStackEntries() || !game.getStack().isEmpty();
        }

        /** Apply {@code edit} and name what changed on the board. */
        Set<String> edit(BoardEdit edit) {
            Map<String, Object> before = BoardStates.of(game);
            edit.apply(game);
            return changes(before, BoardStates.of(game));
        }
    }
}
