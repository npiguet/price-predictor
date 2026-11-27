package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;

import forge.util.MyRandom;

import forge.ai.simulation.GameCopier;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactoryUtil;
import forge.game.combat.Combat;
import forge.game.keyword.Keyword;
import forge.game.spellability.SpellAbility;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.PlayerZoneBattlefield;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * The score check a fork passes before anything is changed in it, on
 * boards whose copies a blank rebuild would score wrong: a copy is only worth recording if it
 * scores as the live game does, so a check that throws loses every fork of
 * that board.
 */
@ExtendWith(ForgeExtension.class)
class ForksScoreCheckTest {

    /**
     * As the declare-blockers step begins an attacker's band is neither
     * blocked nor unblocked yet. The evaluator simulates the combat to come
     * by advancing a copy to combat damage without declaring blockers, and
     * damage assignment must read the unset flag without throwing, or every
     * fork taken at that boundary (702.98a's, 702.70b's) is discarded.
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
     * not among the zone's cards, so the copy's permanent must still name its half
     * for its mana value, the sum of both halves', to be read.
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

    /**
     * Before the second main phase the evaluator scores the combat to come by
     * having the AI declare it in a copy, and the AI's attacks and blocks draw
     * from Forge's one random stream. Scored from that stream, the live game
     * and an exact copy simulate different combats and disagree, so the check
     * scores both from one seeded stream of its own and leaves the live
     * game's stream as it found it.
     */
    @Test
    void theScoreCheckDrawsNothingFromTheLiveGamesRandomStream() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        for (String name : List.of("Grizzly Bears", "Hill Giant", "Craw Wurm")) {
            TestBoard.inPlay(live, you, name).setSickness(false);
        }
        for (String name : List.of("Grizzly Bears", "Hill Giant", "Gray Ogre")) {
            TestBoard.inPlay(live, opponent, name).setSickness(false);
        }
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        CountingRandom stream = new CountingRandom();
        Random was = MyRandom.getRandom();
        MyRandom.setRandom(stream);
        try {
            assertTrue(Forks.scoreCheck(live, copier, fork, you),
                    "an exact copy passes, so the score path ran");
            assertSame(stream, MyRandom.getRandom(), "the live stream was not put back");
        } finally {
            MyRandom.setRandom(was);
        }
        assertEquals(0, stream.draws, "the score check drew from the live game's stream");
    }

    /** A score check that throws is counted under the fork's kind, and discards the fork. */
    @Test
    void aScoreCheckThatThrowsIsCounted() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        GameCopier copier = new GameCopier(live);
        copier.makeCopy();
        String key = "fork score-test score check";
        long before = HookFailures.counts().getOrDefault(key, 0L);

        assertFalse(Forks.scoreCheck("score-test", new Forks.CopyUnderCheck(live, copier, null, you)));

        assertEquals(before + 1, HookFailures.counts().get(key));
    }

    /**
     * A disguised creature is a face-down 2/2 with ward {2}: casting it face
     * down writes the ward into the card's face-down state, which the copier
     * carries over, so the copy's creature keeps its ward and scores as the live game.
     */
    @Test
    void aCopyOfADisguisedCreatureKeepsItsWardAndScoresAsTheLiveGame() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player you = live.getPlayers().get(0);
        Card disguised = TestBoard.inPlay(live, live.getPlayers().get(1), "Alley Assailant");
        SpellAbility faceDown = disguised.getSpellAbilities().stream()
                .filter(sa -> sa.hasParam("FaceDownKeyword")).findFirst().orElseThrow();
        disguised.turnFaceDown(true);
        CardFactoryUtil.setFaceDownState(disguised, faceDown);
        disguised.updateKeywordsCache();
        assertTrue(disguised.hasKeyword(Keyword.WARD), "the live creature has no ward to copy");

        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();

        assertTrue(copier.find(disguised).hasKeyword(Keyword.WARD), "the copy lost the ward");
        assertTrue(Forks.scoreCheck(live, copier, fork, you));
    }

    /**
     * Darkblade Agent has deathtouch as long as its controller has surveilled
     * this turn. The copier carries every count of a player's turn that the cleanup
     * step resets, the surveils included, so the copy's Agent has its
     * deathtouch and scores as the live game.
     */
    @Test
    void aCopyKeepsThePlayersTurnCountsAndWhatReadsThem() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player you = live.getPlayers().get(0);
        Card agent = TestBoard.inPlay(live, you, "Darkblade Agent");
        SpellAbility consider = TestBoard.inHand(live, you, "Consider").getFirstSpellAbility();
        consider.setActivatingPlayer(you);
        live.getAction().moveToLibrary(TestBoard.inHand(live, you, "Island"), consider);
        you.surveil(1, consider, new java.util.HashMap<>());
        you.addInvestigatedThisTurn();
        live.getAction().checkStaticAbilities();
        assertEquals(1, you.getSurveilThisTurn(), "the live player did not surveil");
        assertTrue(agent.hasKeyword(Keyword.DEATHTOUCH), "the live Agent has no deathtouch to copy");

        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Player forkYou = (Player) copier.find(you);

        assertEquals(1, forkYou.getSurveilThisTurn());
        assertEquals(1, forkYou.getInvestigateNumThisTurn());
        assertTrue(copier.find(agent).hasKeyword(Keyword.DEATHTOUCH), "the copy's Agent lost deathtouch");
        assertTrue(Forks.scoreCheck(live, copier, fork, you));
    }

    /** A random stream that counts what is drawn from it. */
    private static final class CountingRandom extends Random {
        int draws;

        CountingRandom() {
            super(1L);
        }

        @Override
        protected int next(int bits) {
            draws++;
            return super.next(bits);
        }
    }
}
