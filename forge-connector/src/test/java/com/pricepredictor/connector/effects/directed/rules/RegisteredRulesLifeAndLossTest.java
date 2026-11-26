package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyAdded;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of life totals: a life exchange (701.12g,
 * 119.7), dethrone (702.105a) and an exchange a can't-gain-life effect
 * stops (119.7). Each rule's scenario base binds, a board missing it binds nothing, and the
 * fork differs from its twin by the variant's one fact, as
 * {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesLifeAndLossTest {

    // ── 701.12g, a life exchange: life ──────────────────────────────────

    private static RuleBoard treeBoard(int opponentLife) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Perdition");
        board.opponent.setLife(opponentLife, null);
        return board;
    }

    @Test
    void lifeExchangeBindsTheTreeAndAnOpponentItWouldNotLower() {
        for (int life : List.of(7, 13)) {
            RuleBoard board = treeBoard(life);
            Binding binding = LifeExchange.bind(board.game).orElseThrow();
            assertEquals(board.subject, binding.card("permanent"));
            assertEquals(board.opponent, binding.player("opponent"));
        }
    }

    @Test
    void lifeExchangeBindsNothingWhenTheExchangeWouldAlreadyLowerTheLife() {
        assertTrue(LifeExchange.bind(treeBoard(20).game).isEmpty());
    }

    /** Under Platinum Emperion (its controller's life total can't change) the exchange does not happen. */
    @Test
    void lifeExchangeBindsNothingWhenTheOpponentCannotLoseLife() {
        RuleBoard board = treeBoard(7);
        board.card(board.opponent, "Platinum Emperion");
        assertTrue(LifeExchange.bind(board.game).isEmpty());
    }

    @Test
    void lifeExchangeForksTheOpponentAboveTheToughness() {
        RuleBoard board = treeBoard(7);
        RuleTwin twin = board.twin(LifeExchange.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(20, twin.fork().player("opponent").getLife());
    }

    // ── 702.105a, dethrone: life ────────────────────────────────────────

    static RuleBoard dethroneBoard(int opponentLife) {
        RuleBoard board = new RuleBoard(PhaseType.COMBAT_DECLARE_ATTACKERS);
        board.subject = board.card(board.you, "Enraged Revolutionary");
        board.you.setLife(15, null);
        board.opponent.setLife(opponentLife, null);
        return board;
    }

    @Test
    void dethroneBindsTheCreatureAndTheOpponentWithMoreLifeThanAnyOther() {
        RuleBoard board = dethroneBoard(20);
        Binding binding = Dethrone.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("attacker"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    /** Tied already, the fork's cut would change nothing; with less life, dethrone would not fire. */
    @Test
    void dethroneBindsNothingAgainstAnOpponentWithNoMoreLifeThanAnother() {
        assertTrue(Dethrone.bind(dethroneBoard(15).game).isEmpty(), "tied");
        assertTrue(Dethrone.bind(dethroneBoard(14).game).isEmpty(), "less");
    }

    /**
     * You at 7 life beside Enraged Revolutionary (2 power, 1 more from its
     * dethrone counter) and a 2-power double striker: counted once the
     * reach is 5, below the 7 the cut opponent keeps, but the double striker
     * deals its 2 twice, and 7 is lethal.
     */
    @Test
    void dethroneCountsADoubleStrikersDamageTwice() {
        RuleBoard board = dethroneBoard(20);
        board.you.setLife(7, null);
        board.card(board.you, "Mirran Crusader");
        assertEquals(7, Dethrone.attackReach(board.you));
        assertTrue(Dethrone.bind(board.game).isEmpty(), "the cut opponent is in the attack's reach");
    }

    @Test
    void dethroneBindsNothingWhenSomethingCanGrowTheAttack() {
        RuleBoard exalted = dethroneBoard(20);
        exalted.card(exalted.you, "Akrasan Squire");
        assertTrue(Dethrone.bind(exalted.game).isEmpty(), "exalted");
        RuleBoard pump = dethroneBoard(20);
        pump.card(pump.you, "Forest");
        TestBoard.inHand(pump.game, pump.you, "Giant Growth");
        assertTrue(Dethrone.bind(pump.game).isEmpty(), "an instant pump it has the land for");
    }

    @Test
    void dethroneForksTheOpponentTiedForTheMostLife() {
        RuleBoard board = dethroneBoard(20);
        RuleTwin twin = board.twin(Dethrone.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(15, twin.fork().player("opponent").getLife());
    }

    // ── 119.7, an exchange a can't-gain-life effect stops: add_permanent

    private static RuleBoard redemptionBoard(boolean leyline) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Redemption");
        if (leyline) {
            board.card(board.you, CantGainLifeExchange.LEYLINE_OF_PUNISHMENT);
        }
        board.you.setLife(5, null);
        return board;
    }

    @Test
    void cantGainLifeExchangeBindsTheTreeAndItsController() {
        RuleBoard board = redemptionBoard(false);
        Binding binding = CantGainLifeExchange.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void cantGainLifeExchangeBindsNothingWhenThePlayerCannotGainLife() {
        assertTrue(CantGainLifeExchange.bind(redemptionBoard(true).game).isEmpty());
    }

    @Test
    void cantGainLifeExchangeBindsNothingWhenTheExchangeWouldLowerTheLife() {
        RuleBoard board = redemptionBoard(false);
        board.you.setLife(13, null);
        assertTrue(CantGainLifeExchange.bind(board.game).isEmpty());
    }

    @Test
    void cantGainLifeExchangeForksLeylineOfPunishment() {
        RuleBoard board = redemptionBoard(false);
        RuleTwin twin = board.twin(CantGainLifeExchange.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), CantGainLifeExchange.LEYLINE_OF_PUNISHMENT);
        assertFalse(twin.fork().player("player").canGainLife());
    }
}
