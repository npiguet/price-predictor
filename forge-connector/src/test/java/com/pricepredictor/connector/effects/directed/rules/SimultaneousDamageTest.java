package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.card.CounterEnumType;
import forge.game.card.token.TokenInfo;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 510.2's precondition binds only a blocker whose Youthful Knight, put in its
 * place, kills the attacker in the first-strike step: the detector tags a
 * creature dealt first-strike damage that dies there having dealt none, and
 * an attacker that survives the Knight's 2 damage shows neither side. The
 * rest of the rule's cases are {@code RegisteredRulesTest}'s.
 */
@ExtendWith(ForgeExtension.class)
class SimultaneousDamageTest {

    /** Your {@code attacker} blocked by the opponent's {@code blocker} as the first-strike damage step begins. */
    private record Blocked(Game game, Card attacker, Card blocker) {

        static Blocked of(String attacker, String blocker) {
            Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
            Player you = game.getPlayers().get(0);
            Player opponent = game.getPlayers().get(1);
            Card attacking = permanent(game, you, attacker);
            Card blocking = permanent(game, opponent, blocker);
            TestBoard.blockedCombat(game, you, opponent, attacking, blocking);
            return new Blocked(game, attacking, blocking);
        }

        private static Card permanent(Game game, Player owner, String name) {
            Card card = TestBoard.inPlay(game, owner, name);
            card.setGameTimestamp(game.getNextTimestamp());
            return card;
        }

        boolean binds() {
            return SimultaneousDamage.bind(game).isPresent();
        }
    }

    /** Hill Giant (3/3) survives the Knight's 2 first-strike damage and kills the Knight. */
    @Test
    void anAttackerThatWouldSurviveTheFirstStrikeDoesNotBind() {
        assertFalse(Blocked.of("Hill Giant", "Grizzly Bears").binds());
    }

    /** The Giant's marked damage leaves it 2 toughness, which the Knight's 2 damage takes. */
    @Test
    void anAttackerWithinReachOfTheKnightsDamageBinds() {
        Blocked board = Blocked.of("Hill Giant", "Grizzly Bears");
        board.attacker().setDamage(1);
        assertTrue(board.binds());
    }

    /** The Bears' +1/+1 counter goes to the Knight in its place, whose 3 power takes the Giant's 3 toughness. */
    @Test
    void whatTheBlockerCarriesCountsTowardTheKnightsDamage() {
        Blocked board = Blocked.of("Hill Giant", "Grizzly Bears");
        board.blocker().setCounters(CounterEnumType.P1P1, 1);
        assertTrue(board.binds());
    }

    /**
     * A Grizzly Bears token blocks the damaged Giant: {@code replace_card}
     * refuses a token, since a card in its place would be a real card, so the
     * board binds nothing where the nontoken Bears binds.
     */
    @Test
    @SuppressWarnings("deprecation")
    void aTokenBlockerDoesNotBind() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = Blocked.permanent(game, you, "Hill Giant");
        giant.setDamage(1);
        Card bears = TestBoard.card(game, opponent, "Grizzly Bears");
        Card copy = new TokenInfo(bears).makeOneToken(opponent, game.nextCardId());
        new CardCopyService(bears).copyCopiableCharacteristics(copy, null, null);
        Card token = game.getAction().moveToPlay(copy, opponent, null, null);
        token.setSickness(false);
        token.setGameTimestamp(game.getNextTimestamp());
        TestBoard.blockedCombat(game, you, opponent, giant, token);
        assertFalse(SimultaneousDamage.bind(game).isPresent());
    }
}
