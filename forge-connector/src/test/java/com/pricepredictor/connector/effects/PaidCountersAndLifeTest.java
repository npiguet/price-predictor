package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A cost half's {@code costs} holds the counters and life its payment moved,
 * from a real payment on a real board: neither survives on the ability, so
 * the bus's events are what say it (finding 9.13, task B14).
 */
@ExtendWith(ForgeExtension.class)
class PaidCountersAndLifeTest {

    @TempDir
    Path tempDir;

    /** The costs of the one cost half the run wrote. */
    private static Map<String, Object> onlyCosts(List<String> all) {
        List<String> halves = records(all, "activation").stream()
                .filter(record -> payload(record).containsKey("costs")).toList();
        assertEquals(1, halves.size(), all.toString());
        return asMap(payload(halves.get(0)).get("costs"));
    }

    /** The only counter the payment moved is loyalty, by {@code expected}. */
    private static void assertLoyalty(int expected, Map<String, Object> costs) {
        Map<String, Object> counters = asMap(costs.get("counters"));
        assertEquals(List.of("LOYALTY"), List.copyOf(counters.keySet()), costs.toString());
        assertEquals(expected, ((Number) counters.get("LOYALTY")).intValue(), costs.toString());
    }

    private static Card withLoyalty(Card walker, Player owner) {
        walker.addCounterInternal(CounterEnumType.LOYALTY, 3, owner, false, null, null);
        return walker;
    }

    private static SpellAbility loyaltyAbility(Card walker, int index) {
        return walker.getSpellAbilities().stream()
                .filter(SpellAbility::isPwAbility).toList().get(index);
    }

    private List<String> activated(Game game, Player activator, SpellAbility ability)
            throws Throwable {
        return ShardRun.observed(tempDir, game, collector -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(activator, ability, null),
                    "the AI activated " + ability);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });
    }

    /** Liliana of the Veil's -2 removes two loyalty counters, and says so. */
    @Test
    void aPlaneswalkersMinusTwoWritesTwoLoyaltyCountersRemoved() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card liliana = withLoyalty(TestBoard.inPlay(game, caster, "Liliana of the Veil"), caster);
        TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility minusTwo = loyaltyAbility(liliana, 1);
        minusTwo.resetTargets();
        minusTwo.getTargets().add(opponent);

        Map<String, Object> costs = onlyCosts(activated(game, caster, minusTwo));

        assertLoyalty(-2, costs);
    }

    /** The +1 is a cost too: a counter put, written positive. */
    @Test
    void aPlaneswalkersPlusOneWritesOneLoyaltyCounterPut() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card liliana = withLoyalty(TestBoard.inPlay(game, caster, "Liliana of the Veil"), caster);
        SpellAbility plusOne = loyaltyAbility(liliana, 0);

        Map<String, Object> costs = onlyCosts(activated(game, caster, plusOne));

        assertLoyalty(1, costs);
    }

    /** Dismember's two Phyrexian symbols are two payments of 2 life: 4 are paid, not the last 2. */
    @Test
    void twoLifePaymentsSum() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Mountain");
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility dismember = TestBoard.spellOf(TestBoard.inHand(game, caster, "Dismember"), caster);
        dismember.resetTargets();
        dismember.getTargets().add(bears);

        Map<String, Object> costs = onlyCosts(activated(game, caster, dismember));

        assertEquals(4, ((Number) costs.get("life")).intValue(), costs.toString());
        assertEquals(16, caster.getLife());
    }
}
