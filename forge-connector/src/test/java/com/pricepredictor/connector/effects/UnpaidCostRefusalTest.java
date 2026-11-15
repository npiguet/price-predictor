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
}
