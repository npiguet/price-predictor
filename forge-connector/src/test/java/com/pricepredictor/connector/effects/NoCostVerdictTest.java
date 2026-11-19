package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A spell whose mana cost is "no cost" (118.6) fails on {@code cost} alone:
 * the catch-all {@code script_restriction} names failures nothing else
 * explains, and the unpayable cost explains this one.
 */
@ExtendWith(ForgeExtension.class)
class NoCostVerdictTest {

    @Test
    void aNoCostSpellInItsOwnersMainPhaseFailsOnCostAlone() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Island");
        SpellAbility vision = TestBoard.inHand(game, a, "Ancestral Vision").getFirstSpellAbility();

        PlayabilityVerdict verdict = PlayabilityVerdict.of(vision, a);

        List<PlayabilityCheck> failed = verdict.failedChecks().stream()
                .map(PlayabilityVerdict.FailedCheck::check).toList();
        assertEquals(List.of(PlayabilityCheck.COST), failed, verdict.toString());
    }
}
