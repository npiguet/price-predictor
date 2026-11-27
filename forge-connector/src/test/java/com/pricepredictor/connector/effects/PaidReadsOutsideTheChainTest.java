package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
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
import java.util.Map;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A line reads what its cost paid also where the read is not in its chain
 *: a permanent spell's {@code K:etbCounter} line
 * counting the colors spent to cast it (converge, adamant), and an ability
 * reading the counters its cost removed. A forced Tajuru Stalwart would enter
 * without its counter and a forced Essence Bottle would remove none.
 */
@ExtendWith(ForgeExtension.class)
class PaidReadsOutsideTheChainTest {

    private static final String REASON = "intervention " + InterventionFork.READS_WHAT_ITS_COST_PAID;

    @TempDir
    Path tempDir;

    private static SpellAbility lineOf(Card card, Player holder, Predicate<SpellAbility> which) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (which.test(ability)) {
                ability.setActivatingPlayer(holder);
                return ability;
            }
        }
        throw new AssertionError(card + " has no such line");
    }

    private void assertRefused(Game game, SpellAbility line) {
        long refused = ForkRefusals.counts().getOrDefault(REASON, 0L);
        assertTrue(InterventionFork.readsAnUnpaidCost(line), line + " reads what its cost paid");

        ShardRun.Fork fork = ShardRun.forked(
                tempDir, new ShardRun.ForkRun(game, TestCaps.caps().interventions(1).probes(0).build(), 1L, line));

        assertFalse(fork.wrote(), "nothing is written for a refused line");
        assertEquals(refused + 1, ForkRefusals.counts().getOrDefault(REASON, 0L), ForkRefusals.counts().toString());
    }

    @Test
    void aPermanentCountingTheColorsSpentToCastItIsRefused() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        assertRefused(game, lineOf(TestBoard.inHand(game, holder, "Tajuru Stalwart"), holder, SpellAbility::isSpell));
    }

    @Test
    void aPermanentWithAnAdamantCounterIsRefused() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        assertRefused(game, lineOf(TestBoard.inHand(game, holder, "Ardenvale Paladin"), holder, SpellAbility::isSpell));
    }

    @Test
    void anAbilityReadingTheCountersItsCostRemovedIsRefused() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card bottle = TestBoard.inPlay(game, holder, "Essence Bottle");
        bottle.addCounterInternal(CounterEnumType.ELIXIR, 3, holder, false, null, Map.of());
        assertRefused(game, lineOf(bottle, holder, line -> line.isActivatedAbility() && line.getApi() != null
                && line.getApi() == forge.game.ability.ApiType.GainLife));
    }

    @Test
    void anAbilityReadingAnSVarItsDieRollCostSetIsRefused() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card golem = TestBoard.inPlay(game, holder, "Clay Golem");
        assertRefused(game, lineOf(golem, holder, SpellAbility::isActivatedAbility));
    }

    @Test
    void aPlainEtbCounterAndAnUnreadCounterCostAreNoRead() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card troll = TestBoard.inHand(game, holder, "Barkhide Troll");
        assertFalse(InterventionFork.readsAnUnpaidCost(lineOf(troll, holder, SpellAbility::isSpell)),
                "an etbCounter of a fixed number reads nothing");
        Card inPlay = TestBoard.inPlay(game, holder, "Barkhide Troll");
        assertFalse(InterventionFork.readsAnUnpaidCost(lineOf(inPlay, holder, SpellAbility::isActivatedAbility)),
                "a SubCounter cost whose ability reads no removed count reads nothing");
    }
}
