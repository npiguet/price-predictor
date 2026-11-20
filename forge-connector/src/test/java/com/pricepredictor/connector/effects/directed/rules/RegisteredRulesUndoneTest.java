package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static com.pricepredictor.connector.effects.directed.rules.CardName.named;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCastingTest.angelBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCastingTest.automatonBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCastingTest.boltBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCastingTest.engineerBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCombatTest.blockedBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCombatTest.unleashBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesCreaturesTest.concertedBoard;
import static com.pricepredictor.connector.effects.directed.rules.RegisteredRulesLifeAndLossTest.dethroneBoard;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What makes a registered rule's variant come undone between the
 * precondition and the fork (review C4, V3 #9-10): a land left to play,
 * a keyword from an attachment or a resolved effect, a sorcery in the
 * non-active hand, carried damage and counters, a defender left in reach.
 * The boards are the rule families' ({@link RegisteredRulesTest} sets out
 * the twin).
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesUndoneTest {

    // ── what makes a variant come undone (review C4, V3 #9-10) ──────────

    /**
     * C4 #2: a land in the holder's hand would be played as the turn goes on
     * and give back the mana a "one land short" variant takes, so the rule
     * does not bind; with the turn's land drop used it binds again.
     */
    private static void assertALandToPlayStopsTheBinding(RuleBoard board, DirectedRule rule) {
        assertTrue(rule.precondition(board.game).isPresent(), rule.name() + ": the base binds");
        TestBoard.inHand(board.game, board.you, "Forest");
        assertTrue(rule.precondition(board.game).isEmpty(), rule.name() + ": binds with a land to play");
        board.you.setLandsPlayedThisTurn(1);
        assertTrue(rule.precondition(board.game).isPresent(), rule.name() + ": the land drop is used");
    }

    @Test
    void prototypeBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(automatonBoard(3), Prototype.rule());
    }

    @Test
    void mayhemBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(boltBoard(true, 2), Mayhem.rule());
    }

    @Test
    void morphFaceUpBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(angelBoard(4), MorphFaceUp.rule());
    }

    @Test
    void improviseBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(engineerBoard(3), Improvise.rule());
    }

    /** A modal double-faced card whose back is a land is a land to play too. */
    @Test
    void aModalCardWithALandBackIsALandToPlay() {
        RuleBoard board = automatonBoard(3);
        TestBoard.inHand(board.game, board.you, "Turntimber Symbiosis");
        assertTrue(Prototype.bind(board.game).isEmpty());
    }

    /**
     * C4 #4: a keyword the bound creature has from an Equipment stays on the
     * card {@code replace_card} puts in its place, since the edit relinks the
     * attachment; so does one a resolved effect gave it.
     */
    /** Sword of Fire and Ice (protection from red and from blue) on {@code creature}. */
    private static void equipSword(RuleBoard board, Card creature) {
        Card sword = board.card(creature.getController(), "Sword of Fire and Ice");
        sword.attachToEntity(creature, null);
        board.game.getAction().checkStaticAbilities();
        assertTrue(creature.hasKeyword(Keyword.PROTECTION));
    }

    @Test
    void sharedKeywordsBindsNothingWhenTheKeywordComesFromAnAttachment() {
        RuleBoard board = concertedBoard("Grizzly Bears", "Hill Giant");
        equipSword(board, board.subject);
        assertTrue(SharedKeywords.bind(board.game).isEmpty());
    }

    @Test
    void sharedKeywordsBindsNothingWhenTheKeywordComesFromAResolvedEffect() {
        RuleBoard board = concertedBoard("Grizzly Bears", "Hill Giant");
        KeywordEdit.add(board.game, board.subject, "First Strike");
        assertTrue(board.subject.hasFirstStrike());
        assertTrue(SharedKeywords.bind(board.game).isEmpty());
    }

    /**
     * C4 #5: a sorcery in the non-active player's hand cannot be cast this
     * turn, so the fork would be spent on nothing. An instant can
     * ({@code toughnessZeroBindsTheCreatureTheSpellWouldKill}'s Disfigure,
     * {@code limitedSearchBindsTheLimitingPermanent}'s Vampiric Tutor).
     */
    @Test
    void toughnessZeroBindsNothingForASorceryInTheNonActiveHand() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Drudge Skeletons");
        for (int i = 0; i < 3; i++) {
            board.card(board.opponent, "Swamp");
        }
        Card fatigue = TestBoard.inHand(board.game, board.opponent, "Crippling Fatigue");
        assertEquals(2, ToughnessZero.toughnessLoss(fatigue));
        assertTrue(ToughnessZero.bind(board.game).isEmpty());

        RuleBoard instant = new RuleBoard(PhaseType.MAIN1);
        instant.card(instant.you, "Drudge Skeletons");
        instant.card(instant.opponent, "Swamp");
        TestBoard.inHand(instant.game, instant.opponent, "Disfigure");
        assertTrue(ToughnessZero.bind(instant.game).isPresent(), "Disfigure is an instant");
    }

    @Test
    void smallLibraryShuffleBindsNothingForASorceryInTheNonActiveHand() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Cosi's Trickster");
        board.card(board.opponent, "Swamp");
        board.card(board.opponent, "Swamp");
        TestBoard.inHand(board.game, board.opponent, "Demonic Tutor");
        board.library(board.opponent, "Forest", "Grizzly Bears");
        assertTrue(SmallLibraryShuffle.bind(board.game).isEmpty());
    }

    /**
     * C4 #1, V3 #9 (b): {@code replace_card} carries the permanent's marked
     * damage and counters to the card put in its place, so a smaller card
     * dies of them at the fork's first state-based check.
     */
    @Test
    void simultaneousDamageBindsNothingWhenTheBlockerCarriesDamage() {
        RuleBoard board = blockedBoard(named("Grizzly Bears"));
        board.subject.setDamage(1);
        assertTrue(SimultaneousDamage.bind(board.game).isEmpty(), "Youthful Knight is a 2/1");
    }

    /** Serra Angel (4/4) lives with 3 damage; the Hill Giant (3/3) put in its place would not. */
    @Test
    void sharedKeywordsBindsNothingWhenTheHolderCarriesDamage() {
        RuleBoard board = concertedBoard("Serra Angel", "Grizzly Bears");
        assertTrue(SharedKeywords.bind(board.game).isPresent(), "Serra Angel has flying and vigilance");
        board.subject.setDamage(3);
        assertTrue(SharedKeywords.bind(board.game).isEmpty(), "Hill Giant is a 3/3");
    }

    /** C4 #6: taking the counter away kills a creature whose toughness rests on it. */
    @Test
    void unleashBindsNothingWhenTheCreatureLivesByItsCounter() {
        RuleBoard board = unleashBoard(named("Rakdos Cackler"), 1);
        board.subject.setDamage(2);
        assertTrue(UnleashCantBlock.bind(board.game).isEmpty(), "a 2/2 Cackler with 2 damage dies at 1 toughness");
    }

    /**
     * C4 #8: the defender cut to the most anyone else has must not be left
     * in reach of the attack the fork's point is.
     */
    @Test
    void dethroneBindsNothingWhenTheCutWouldLeaveTheDefenderInReach() {
        RuleBoard board = dethroneBoard(20);
        board.you.setLife(3, null);
        assertTrue(Dethrone.bind(board.game).isEmpty(), "3 life against Enraged Revolutionary's 2 power and its dethrone counter");
        RuleBoard safe = dethroneBoard(20);
        safe.you.setLife(4, null);
        assertTrue(Dethrone.bind(safe.game).isPresent());
    }
}
