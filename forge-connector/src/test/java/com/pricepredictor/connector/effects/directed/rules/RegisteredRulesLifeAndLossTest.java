package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyAdded;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of life totals and losing the game: a life
 * exchange (701.12g, 119.7), dethrone (702.105a), a win at a life total
 * (104.2b), ten poison counters (104.3d), a draw from an empty library
 * (704.5b) and a draw when every player loses at once (104.4a). Each
 * rule's scenario base binds, a board missing it binds nothing, and the
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

    // ── 104.2b, a win at a life total: life ─────────────────────────────

    private static RuleBoard sovereignBoard(int life) {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Felidar Sovereign");
        board.you.setLife(life, null);
        return board;
    }

    @Test
    void lifeTotalWinBindsTheSovereignBelowFortyLife() {
        RuleBoard board = sovereignBoard(39);
        Binding binding = LifeTotalWin.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void lifeTotalWinBindsNothingAtFortyLife() {
        assertTrue(LifeTotalWin.bind(sovereignBoard(40).game).isEmpty());
    }

    /** Platinum Emperion stops the life change the variant makes. */
    @Test
    void lifeTotalWinBindsNothingWhenTheLifeTotalCannotChange() {
        RuleBoard board = sovereignBoard(39);
        board.card(board.you, "Platinum Emperion");
        assertTrue(LifeTotalWin.bind(board.game).isEmpty());
    }

    @Test
    void lifeTotalWinForksThePlayerAtFortyLife() {
        RuleTwin twin = sovereignBoard(12).twin(LifeTotalWin.rule());
        assertEquals(Set.of("P0.life"), twin.changes());
        assertEquals(40, twin.fork().player("player").getLife());
    }

    // ── 104.3d, ten poison counters: counters ───────────────────────────

    /** B at {@code poison} counters; A holds Ichor Rats over three Swamps. */
    private static RuleBoard poisonBoard(int poison) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Swamp");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Ichor Rats");
        board.opponent.setCounters(CounterEnumType.POISON, poison, null, false);
        return board;
    }

    @Test
    void poisonLossBindsAnOpponentAtAnyPoisonBelowNineAndThePoisonSource() {
        RuleBoard board = poisonBoard(0);
        Binding binding = PoisonLoss.bind(board.game).orElseThrow();
        assertEquals(board.opponent, binding.player("player"));
        assertEquals(board.other, binding.card("card"));
        assertEquals(board.you, binding.player("caster"));
    }

    @Test
    void poisonLossBindsNothingWithoutAPoisonSourceToCast() {
        RuleBoard board = poisonBoard(0);
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(PoisonLoss.bind(board.game).isEmpty());
    }

    @Test
    void poisonLossBindsNothingOneCounterShort() {
        assertTrue(PoisonLoss.bind(poisonBoard(9).game).isEmpty());
    }

    /** The caster's own poison is no matter: the opponent's is the one the edit sets. */
    @Test
    void poisonLossBindsTheOpponentNotTheCaster() {
        RuleBoard board = poisonBoard(3);
        board.you.setCounters(CounterEnumType.POISON, 5, null, false);
        assertEquals(board.opponent, PoisonLoss.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void poisonLossForksThePlayerWithNineCounters() {
        RuleTwin twin = poisonBoard(0).twin(PoisonLoss.rule());
        assertEquals(Set.of("P1.poison"), twin.changes());
        assertEquals(9, twin.fork().player("player").getPoisonCounters());
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

    // ── 704.5b (121.4), a draw step's draw from an empty library: move_card

    /** Your upkeep of game turn {@code turn}, {@code library} Forests in your library. */
    private static RuleBoard lastCardBoard(int library, PhaseType phase, int turn) {
        RuleBoard board = new RuleBoard(phase);
        board.game.getPhaseHandler().devModeSet(phase, board.you, turn);
        board.card(board.you, "Island");
        for (int i = 0; i < library; i++) {
            board.subject = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, "Forest"), -1, null, null);
        }
        return board;
    }

    @Test
    void drawFromEmptyLibraryBindsTheLastCardAsTheUpkeepBegins() {
        RuleBoard board = lastCardBoard(1, PhaseType.UPKEEP, 3);
        assertEquals(board.subject, DrawFromEmptyLibrary.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void drawFromEmptyLibraryBindsNothingWithTwoCardsLeft() {
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(2, PhaseType.UPKEEP, 3).game).isEmpty());
    }

    @Test
    void drawFromEmptyLibraryBindsNothingWhereNoDrawStepFollows() {
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(1, PhaseType.MAIN1, 3).game).isEmpty(),
                "past the draw step");
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(1, PhaseType.UPKEEP, 1).game).isEmpty(),
                "the first turn skips its draw step");
    }

    @Test
    void drawFromEmptyLibraryForksAnEmptyLibrary() {
        RuleBoard board = lastCardBoard(1, PhaseType.UPKEEP, 3);
        RuleTwin twin = board.twin(DrawFromEmptyLibrary.rule());
        String forest = id(board.subject);
        assertTrue(twin.changes().containsAll(Set.of("P0.library", "P0.graveyard")), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.library", "P0.graveyard").contains(change) || change.startsWith(forest + "."),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library).isEmpty());
    }

    // ── 104.4a, a draw when every player loses at once: life, inverted ──

    /** Your Plague Spitter as your upkeep begins, you at {@code yours} life and the opponent at {@code theirs}. */
    private static RuleBoard spitterBoard(int yours, int theirs) {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Plague Spitter");
        board.you.setLife(yours, null);
        board.opponent.setLife(theirs, null);
        return board;
    }

    @Test
    void simultaneousLossBindsAnUpkeepTriggerTakingLifeFromEachPlayerAtAnyLife() {
        RuleBoard board = spitterBoard(20, 20);
        Binding binding = SimultaneousLoss.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.opponent, binding.player("opponent"));
        assertTrue(SimultaneousLoss.bind(spitterBoard(1, 20).game).isPresent(), "one player in reach already");
    }

    /** Every player within the trigger's reach is the rare board itself: the edit would add nothing to it. */
    @Test
    void simultaneousLossBindsNothingWhenEveryPlayerIsInReachAlready() {
        assertTrue(SimultaneousLoss.bind(spitterBoard(1, 1).game).isEmpty());
    }

    @Test
    void simultaneousLossBindsNothingForTheOtherPlayersTrigger() {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.card(board.opponent, "Plague Spitter");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Stormfist Crusader's chain draws, then takes 1 life from each player. */
    @Test
    void simultaneousLossReadsTheLifeLossBelowADraw() {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.card(board.you, "Stormfist Crusader");
        assertTrue(SimultaneousLoss.bind(board.game).isPresent());
    }

    @Test
    void simultaneousLossBindsNothingAtAnotherBoundary() {
        RuleBoard board = new RuleBoard(PhaseType.DRAW);
        board.card(board.you, "Plague Spitter");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Emperion keeps its controller's life from changing: the trigger would leave the opponent at 20. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseLife() {
        RuleBoard board = spitterBoard(20, 20);
        board.card(board.opponent, "Platinum Emperion");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Angel's controller cannot lose the game: the 0 life loses it nothing. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseForZeroLife() {
        RuleBoard board = spitterBoard(20, 20);
        board.card(board.opponent, "Platinum Angel");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** A trigger asking a condition of its own, or its controller whether it happens, is no engine certainty. */
    @Test
    void simultaneousLossBindsNothingForATriggerWithAConditionOrAChoice() {
        for (Map.Entry<String, String> param : Map.of("IsPresent", "Card.Self", "CheckSVar", "X",
                "OptionalDecider", "You").entrySet()) {
            RuleBoard board = spitterBoard(20, 20);
            upkeepTrigger(board.subject).putParam(param.getKey(), param.getValue());
            assertTrue(SimultaneousLoss.bind(board.game).isEmpty(), param.getKey());
        }
    }

    /** A chain segment with a condition, an unless cost or an optional part, or another API, decides nothing for sure. */
    @Test
    void simultaneousLossBindsNothingForAChainWithAConditionOrAnotherApi() {
        String damage = "DB$ DamageAll | ValidCards$ Creature | ValidPlayers$ Player | NumDmg$ 1";
        for (String extra : List.of(" | ConditionCheckSVar$ X", " | UnlessCost$ 1", " | Optional$ True",
                " | SubAbility$ DBDestroy")) {
            RuleBoard board = spitterBoard(20, 20);
            board.subject.setSVar("TrigDamageAll", damage + extra);
            board.subject.setSVar("DBDestroy", "DB$ Destroy | Defined$ Self");
            assertTrue(SimultaneousLoss.bind(board.game).isEmpty(), extra);
        }
        RuleBoard plain = spitterBoard(20, 20);
        plain.subject.setSVar("TrigDamageAll", damage);
        assertTrue(SimultaneousLoss.bind(plain.game).isPresent(), "the plain chain binds");
    }

    private static forge.game.trigger.Trigger upkeepTrigger(Card card) {
        return card.getTriggers().stream()
                .filter(trigger -> "Upkeep".equals(trigger.getParam("Phase"))).findFirst().orElseThrow();
    }

    @Test
    void simultaneousLossForksEachPlayerAtTheTriggersAmount() {
        RuleTwin twin = spitterBoard(20, 20).twin(SimultaneousLoss.rule());
        assertEquals(Set.of("P0.life", "P1.life"), twin.changes());
        assertEquals(1, twin.fork().player("player").getLife());
        assertEquals(1, twin.fork().player("opponent").getLife());
    }

    /** A player already in the trigger's reach keeps its life: the edit only lowers a life total. */
    @Test
    void simultaneousLossForksOnlyThePlayerOutOfReach() {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.card(board.you, "Plague Spitter");
        board.you.setLife(1, null);
        RuleTwin twin = board.twin(SimultaneousLoss.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(1, twin.fork().player("opponent").getLife());
    }
}
