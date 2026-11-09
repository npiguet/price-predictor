package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import forge.ai.simulation.GameCopier;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.PlayerZoneBattlefield;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * The score check a fork passes before anything is changed in it, on the
 * boards whose copies it used to refuse: a copy is only worth recording if it
 * scores as the live game does, so a check that throws loses every fork of
 * that board.
 */
@ExtendWith(ForgeExtension.class)
class ForksScoreCheckTest {

    /**
     * As the declare-blockers step begins an attacker's band is neither
     * blocked nor unblocked yet. The evaluator simulates the combat to come
     * by advancing a copy to combat damage without declaring blockers, so
     * damage assignment read the unset flag and threw, and every fork taken
     * at that boundary (702.98a's, 702.70b's) was discarded.
     */
    @Test
    void aCopyAtDeclareBlockersBeforeAnyBlockerScoresAsTheLiveGame() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(live, you, "Grizzly Bears");
        TestBoard.inPlay(live, opponent, "Rakdos Cackler");
        Combat combat = new Combat(you);
        combat.addAttacker(attacker, opponent);
        live.getPhaseHandler().setCombat(combat);

        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();

        assertTrue(Forks.scoreCheck(live, copier, fork, you));
    }

    /**
     * A melded permanent's other half sits in its battlefield's melded list,
     * not among the zone's cards, so the copy's permanent named no half and
     * its mana value, the sum of both halves', threw on the copy.
     */
    @Test
    void aCopyWithAMeldedPermanentKeepsItsOtherHalfAndScoresAsTheLiveGame() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Card host = TestBoard.inPlay(live, you, "Graf Rats");
        Card half = TestBoard.inHand(live, you, "Midnight Scavengers");
        host.changeToState(CardStateName.Meld);
        host.setBackSide(true);
        host.setMeldedWith(half);
        ((PlayerZoneBattlefield) you.getZone(ZoneType.Battlefield)).addToMelded(half);

        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Card forkedHost = copier.find(host);

        assertNotNull(forkedHost.getMeldedWith(), "the copy names the other half");
        assertEquals("Midnight Scavengers", forkedHost.getMeldedWith().getName());
        assertEquals(host.getCMC(), forkedHost.getCMC());
        assertTrue(Forks.scoreCheck(live, copier, fork, you));
    }
}
