package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An intervention never forces a line whose resolution reads a cost the fork
 * did not pay where Forge reads it unguarded: ninjutsu's resolution takes
 * the attacker returned to hand off its paid list, which a forced line
 * never filled, and threw a {@code NullPointerException} in
 * {@code ChangeZoneEffect.changeKnownOriginResolve}. It is refused, counted
 * as {@code intervention reads a cost it never paid}, before it is forced.
 *
 * <p>Crewing and saddling read their paid lists too, but through a guard:
 * a forced crew or saddle, the first of the turn or a second, resolves.
 */
@ExtendWith(ForgeExtension.class)
class UnpaidCostRefusalTest {

    private static final String REFUSAL = "intervention " + InterventionFork.READS_AN_UNPAID_COST;

    @TempDir
    Path tempDir;

    private static SpellAbility ninjutsuOf(Card card, Player holder) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (ability.isNinjutsu()) {
                ability.setActivatingPlayer(holder);
                return ability;
            }
        }
        throw new AssertionError(card + " has no ninjutsu");
    }

    @Test
    void ninjutsuIsRefusedBeforeItIsForced() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card ninja = TestBoard.inHand(game, holder, "Ninja of the Deep Hours");
        SpellAbility ninjutsu = ninjutsuOf(ninja, holder);
        long refused = ForkRefusals.counts().getOrDefault(REFUSAL, 0L);
        long thrown = HookFailures.counts().getOrDefault("fork intervention body", 0L);

        ShardRun.Fork fork = ShardRun.forked(
                tempDir, game, TestCaps.caps().interventions(1).probes(0).build(), 1L, ninjutsu);

        assertEquals(thrown, HookFailures.counts().getOrDefault("fork intervention body", 0L),
                "the forced body threw: " + HookFailures.counts());
        assertFalse(fork.wrote(), "nothing is written for a refused line");
        assertEquals(refused + 1, ForkRefusals.counts().getOrDefault(REFUSAL, 0L),
                ForkRefusals.counts().toString());
        assertTrue(game.getCardState(ninja).isInZone(forge.game.zone.ZoneType.Hand), "the live ninja stays in hand");
        assertTrue(InterventionFork.readsAnUnpaidCost(ninjutsu), "ninjutsu reads its returned attacker");
    }

    @Test
    void anOrdinaryLineReadsNoUnpaidCost() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card ninja = TestBoard.inHand(game, holder, "Ninja of the Deep Hours");
        SpellAbility cast = ninja.getFirstSpellAbility();
        cast.setActivatingPlayer(holder);

        assertFalse(InterventionFork.readsAnUnpaidCost(cast), "casting the ninja reads no cost");
    }

    private static SpellAbility lineOf(Card card, Player holder, java.util.function.Predicate<SpellAbility> which) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (which.test(ability)) {
                ability.setActivatingPlayer(holder);
                return ability;
            }
        }
        throw new AssertionError(card + " has no such line");
    }

    /** Force {@code line} in a fork; assert it neither threw nor was refused. */
    private void forcedWithoutRefusal(Game game, SpellAbility line) {
        long refused = ForkRefusals.counts().getOrDefault(REFUSAL, 0L);
        long thrown = HookFailures.counts().getOrDefault("fork intervention body", 0L);

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, TestCaps.caps().interventions(1).probes(0).build(), 1L, line,
                "to tell a forced line that resolved and did nothing from one that never resolved");

        assertEquals(thrown, HookFailures.counts().getOrDefault("fork intervention body", 0L),
                "the forced body threw: " + HookFailures.counts());
        assertEquals(refused, ForkRefusals.counts().getOrDefault(REFUSAL, 0L), ForkRefusals.counts().toString());
        assertTrue(fork.wrote(), "the forced line resolved and was written: " + ForkRefusals.counts());
    }

    @Test
    void aNeverCrewedVehiclesCrewIsForced() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card copter = TestBoard.inPlay(game, holder, "Smuggler's Copter");
        TestBoard.inPlay(game, holder, "Hill Giant");
        SpellAbility crew = lineOf(copter, holder, SpellAbility::isCrew);

        assertFalse(InterventionFork.readsAnUnpaidCost(crew), "a vehicle not crewed this turn holds no crew list");
        forcedWithoutRefusal(game, crew);
    }

    @Test
    void aVehicleCrewedThisTurnIsCrewedAgain() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card copter = TestBoard.inPlay(game, holder, "Smuggler's Copter");
        Card giant = TestBoard.inPlay(game, holder, "Hill Giant");
        copter.addCrewedByThisTurn(new forge.game.card.CardCollection(giant));
        SpellAbility crew = lineOf(copter, holder, SpellAbility::isCrew);

        assertFalse(InterventionFork.readsAnUnpaidCost(crew), "a second crew adds a null paid list as nothing");
        forcedWithoutRefusal(game, crew);
    }

    @Test
    void aMountSaddledThisTurnIsSaddledAgain() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card jaguar = TestBoard.inPlay(game, holder, "Alacrian Jaguar");
        Card giant = TestBoard.inPlay(game, holder, "Hill Giant");
        jaguar.setSaddled(true);
        jaguar.addSaddledByThisTurn(new forge.game.card.CardCollection(giant));
        SpellAbility saddle = lineOf(jaguar, holder, line -> line.isKeyword(forge.game.keyword.Keyword.SADDLE));

        assertFalse(InterventionFork.readsAnUnpaidCost(saddle), "a second saddle adds a null paid list as nothing");
        forcedWithoutRefusal(game, saddle);
    }
}
