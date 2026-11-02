package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtilCost;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The rules' verdict on a candidate, on boards where the AI's answer and the
 * rules' answer disagree — the only boards where a test can tell them apart —
 * and the checks it names as failed (FR-031, FR-032).
 */
@ExtendWith(ForgeExtension.class)
class PlayabilityVerdictTest {

    private static SpellAbility abilityOn(Card host, String script) {
        return AbilityFactory.getAbility(script, host);
    }

    /** No target has been chosen, which is what the AI leaves when it stops early; the rules still offer one. */
    @Test
    void aTargetTheRulesOfferCountsWhetherOrNotOneWasChosen() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility pump = abilityOn(bears, "AB$ Pump | Cost$ 0 | ValidTgts$ Creature | NumAtt$ +1");

        assertTrue(pump.getTargets().isEmpty(), "the board under test has no chosen target");
        assertTrue(PlayabilityVerdict.of(pump, a).hasLegalTarget());
    }

    /** And a target the rules do not offer is absent however the ability was left. */
    @Test
    void aTargetTheRulesDoNotOfferIsAbsent() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility destroy = abilityOn(bears, "AB$ Destroy | Cost$ 0 | ValidTgts$ Artifact");

        assertFalse(PlayabilityVerdict.of(destroy, a).hasLegalTarget());
    }

    /** Fiery Intervention: 5 damage to target creature, or destroy target artifact. */
    private static SpellAbility fieryIntervention(Game game, Player owner) {
        return TestBoard.inHand(game, owner, "Fiery Intervention").getFirstSpellAbility();
    }

    /** One choosable mode is all a choose-one charm asks for, and none is too few. */
    @Test
    void aCharmIsPlayableWhenEnoughOfItsModesCanBeChosen() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility charm = fieryIntervention(game, a);

        assertFalse(PlayabilityVerdict.of(charm, a).hasLegalTarget(),
                "no creature and no artifact: neither mode can be chosen");
        TestBoard.inPlay(game, a, "Grizzly Bears");
        assertTrue(PlayabilityVerdict.of(charm, a).hasLegalTarget(),
                "a creature makes the damage mode choosable");
    }

    /**
     * A charm's chain is whatever modes someone chained onto it last; a stale
     * chain naming the untargetable mode does not make the charm unplayable.
     */
    @Test
    void aCharmIsJudgedByItsOfferedModesNotItsLeftoverChain() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility charm = fieryIntervention(game, a);
        charm.setSubAbility(charm.getAdditionalAbilityList("Choices").stream()
                .filter(mode -> mode.getApi() == ApiType.Destroy).findFirst().orElseThrow());

        assertTrue(PlayabilityVerdict.of(charm, a).hasLegalTarget());
    }

    /**
     * The AI's cost check refuses a planeswalker's last loyalty when the
     * opponent has no creature; the rules let it be paid.
     */
    @Test
    void affordabilityIgnoresTheAisReluctanceToSpendAPlaneswalker() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card jace = TestBoard.inPlay(game, a, "Jace Beleren");
        jace.setCounters(CounterEnumType.LOYALTY, 1);
        SpellAbility minusOne = jace.getSpellAbilities().stream()
                .filter(ability -> ability.isPwAbility()
                        && ability.getPayCosts().toSimpleString().contains("1"))
                .filter(ability -> ability.getApi() == ApiType.Draw && ability.usesTargeting())
                .findFirst().orElseThrow();
        minusOne.setActivatingPlayer(a);

        assertFalse(ComputerUtilCost.canPayCost(minusOne, a, false),
                "the board under test is one the AI's own check refuses");
        assertTrue(PlayabilityVerdict.of(minusOne, a).affordable());
    }

    /**
     * The verdict is for the player it names, and leaves the candidate as it
     * found it: a candidate with no activating player still has none.
     */
    @Test
    void aVerdictIsForThePlayerItNamesAndLeavesTheActivatorAsItWas() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player b = game.getPlayers().get(1);
        SpellAbility pump = abilityOn(TestBoard.inPlay(game, b, "Grizzly Bears"),
                "AB$ Pump | Cost$ 0 | NumAtt$ +1");

        assertTrue(PlayabilityVerdict.of(pump, b).canPlay(),
                "an instant-speed ability of the player whose turn it is not");
        assertNull(pump.getActivatingPlayer());
    }

    // ── timing and failed checks ────────────────────────────────────────

    private static List<PlayabilityCheck> checks(PlayabilityVerdict verdict) {
        return verdict.failedChecks().stream().map(PlayabilityVerdict.FailedCheck::check).toList();
    }

    /**
     * Acceptance 8 on one candidate: a sorcery in the hand of the player
     * whose turn it is not fails timing, and timing alone.
     */
    @Test
    void aSorceryOnTheOpponentsTurnFailsCastingSpeed() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player b = game.getPlayers().get(1);
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inPlay(game, b, "Island");
        SpellAbility divination = TestBoard.inHand(game, b, "Divination").getFirstSpellAbility();

        PlayabilityVerdict verdict = PlayabilityVerdict.of(divination, b);

        assertFalse(verdict.timing());
        assertFalse(verdict.canPlay());
        assertTrue(verdict.affordable(), "the islands pay for it; only timing is wrong");
        assertEquals(List.of(PlayabilityCheck.CASTING_SPEED), checks(verdict));
    }

    @Test
    void aSorceryInItsOwnersMainPhaseFailsNothing() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Island");
        TestBoard.inPlay(game, a, "Island");
        TestBoard.inPlay(game, a, "Island");
        SpellAbility divination = TestBoard.inHand(game, a, "Divination").getFirstSpellAbility();

        PlayabilityVerdict verdict = PlayabilityVerdict.of(divination, a);

        assertTrue(verdict.timing());
        assertTrue(verdict.canPlay());
        assertEquals(List.of(), verdict.failedChecks());
    }

    /** A split-second spell on the stack stops an instant, whose timing is fine. */
    @Test
    void aSplitSecondSpellOnTheStackFailsSplitSecond() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        TestBoard.inPlay(game, b, "Mountain");
        SpellAbility bolt = TestBoard.inHand(game, b, "Lightning Bolt").getFirstSpellAbility();
        TestBoard.cast(game, TestBoard.inHand(game, a, "Sudden Shock"), a, b);
        assertTrue(game.getStack().isSplitSecondOnStack(), "the board under test");

        PlayabilityVerdict verdict = PlayabilityVerdict.of(bolt, b);

        assertTrue(verdict.timing(), "an instant");
        assertFalse(verdict.canPlay());
        assertEquals(List.of(PlayabilityCheck.SPLIT_SECOND), checks(verdict));
    }

    private static StaticAbility staticOf(Card card, StaticAbilityMode mode) {
        return card.getStaticAbilities().stream()
                .filter(st -> st.checkMode(mode)).findFirst().orElseThrow();
    }

    /**
     * Grand Abolisher forbids the opponent's instant on its controller's turn:
     * Forge's {@code canPlay} allows it, the static does not, and the check
     * names the static.
     */
    @Test
    void aStaticProhibitionNamesItsStatic() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card abolisher = TestBoard.inPlay(game, a, "Grand Abolisher");
        TestBoard.inPlay(game, b, "Mountain");
        SpellAbility bolt = TestBoard.inHand(game, b, "Lightning Bolt").getFirstSpellAbility();
        ProvenanceKey cantBeCast = ProvenanceResolver.of(staticOf(abolisher, StaticAbilityMode.CantBeCast));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(bolt, b);

        assertTrue(verdict.timing());
        assertFalse(verdict.canPlay());
        assertEquals(List.of(new PlayabilityVerdict.FailedCheck(
                PlayabilityCheck.STATIC_PROHIBITION, cantBeCast)), verdict.failedChecks());
        assertEquals(cantBeCast, verdict.prohibitingStatic());
    }

    /** Its other static stops an opponent's creature ability the same way. */
    @Test
    void anActivationProhibitionNamesItsStatic() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card abolisher = TestBoard.inPlay(game, a, "Grand Abolisher");
        SpellAbility pump = abilityOn(TestBoard.inPlay(game, b, "Grizzly Bears"),
                "AB$ Pump | Cost$ 0 | NumAtt$ +1");

        PlayabilityVerdict verdict = PlayabilityVerdict.of(pump, b);

        assertFalse(verdict.canPlay());
        assertEquals(ProvenanceResolver.of(staticOf(abolisher, StaticAbilityMode.CantBeActivated)),
                verdict.prohibitingStatic());
    }

    /**
     * A legendary sorcery with no legendary permanent fails a check Forge
     * evaluates among a dozen others and does not name.
     */
    @Test
    void aBundledRestrictionReportsScriptRestriction() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility blast = TestBoard.inHand(game, a, "Urza's Ruinous Blast").getFirstSpellAbility();

        PlayabilityVerdict verdict = PlayabilityVerdict.of(blast, a);

        assertFalse(verdict.canPlay());
        assertTrue(verdict.failed(PlayabilityCheck.SCRIPT_RESTRICTION), verdict.toString());
        assertFalse(verdict.failed(PlayabilityCheck.SCRIPT_CONDITION), verdict.toString());
    }

    /** {@code Activation$ Threshold} is the script's own condition on the board. */
    @Test
    void anUnmetActivationConditionReportsScriptCondition() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility threshold = abilityOn(TestBoard.inPlay(game, a, "Grizzly Bears"),
                "AB$ Pump | Cost$ 0 | Activation$ Threshold | NumAtt$ +1");

        PlayabilityVerdict verdict = PlayabilityVerdict.of(threshold, a);

        assertFalse(verdict.canPlay());
        assertEquals(List.of(PlayabilityCheck.SCRIPT_CONDITION), checks(verdict));
    }

    @Test
    void aSpentActivationLimitReportsActivationLimit() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility once = abilityOn(bears, "AB$ Pump | Cost$ 0 | ActivationLimit$ 1 | NumAtt$ +1");
        bears.addAbilityActivated(once);

        PlayabilityVerdict verdict = PlayabilityVerdict.of(once, a);

        assertFalse(verdict.canPlay());
        assertEquals(List.of(PlayabilityCheck.ACTIVATION_LIMIT), checks(verdict));
    }

    /** A loyalty ability after one this turn: a limit Forge checks among its other restrictions. */
    @Test
    void aSecondLoyaltyAbilityReportsActivationLimit() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card jace = TestBoard.inPlay(game, a, "Jace Beleren");
        jace.setCounters(CounterEnumType.LOYALTY, 3);
        jace.addPlaneswalkerAbilityActivated();
        SpellAbility loyalty = jace.getSpellAbilities().stream()
                .filter(SpellAbility::isPwAbility).findFirst().orElseThrow();

        PlayabilityVerdict verdict = PlayabilityVerdict.of(loyalty, a);

        assertFalse(verdict.canPlay());
        assertEquals(List.of(PlayabilityCheck.ACTIVATION_LIMIT), checks(verdict));
    }

    /** An ability whose mana its player lacks fails the cost check and nothing else. */
    @Test
    void anUnaffordableAbilityReportsCost() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility pump = abilityOn(TestBoard.inPlay(game, a, "Grizzly Bears"),
                "AB$ Pump | Cost$ 5 | NumAtt$ +1");

        PlayabilityVerdict verdict = PlayabilityVerdict.of(pump, a);

        assertTrue(verdict.canPlay(), "cost is its own bit, not part of can_play");
        assertFalse(verdict.affordable());
        assertEquals(List.of(PlayabilityCheck.COST), checks(verdict));
    }

    /** Prey Upon targets a creature on each side: the record lists both. */
    @Test
    void aSpellWithTwoTargetingClausesListsTheirUnion() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, a, "Grizzly Bears");
        Card theirs = TestBoard.inPlay(game, b, "Hill Giant");
        SpellAbility preyUpon = TestBoard.inHand(game, a, "Prey Upon").getFirstSpellAbility();

        DecisionCandidate read = DecisionCandidate.read(preyUpon, a).orElseThrow();

        assertEquals(List.of(Refs.of(mine), Refs.of(theirs)), read.legalTargets());
    }

    // ── land plays (FR-033a) ────────────────────────────────────────────

    private static SpellAbility landPlayOf(Card land) {
        return land.getSpellAbilities().stream()
                .filter(SpellAbility::isLandAbility).findFirst().orElseThrow();
    }

    /** A land in hand on its owner's turn, the drop unused: playable, nothing failed. */
    @Test
    void aLandInHandOnItsOwnersTurnFailsNothing() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility forest = landPlayOf(TestBoard.inHand(game, a, "Forest"));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(forest, a);

        assertTrue(verdict.canPlay());
        assertTrue(verdict.timing());
        assertTrue(verdict.affordable() && verdict.hasLegalTarget(), "a land costs and targets nothing");
        assertEquals(List.of(), verdict.failedChecks());
        assertNull(forest.getActivatingPlayer(), "the verdict puts the activator back");
    }

    /** A second land in one turn fails the land-play limit, not the activation limit. */
    @Test
    void aSecondLandFailsTheLandPlayLimit() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        a.playLand(TestBoard.inHand(game, a, "Forest"), null);
        SpellAbility mountain = landPlayOf(TestBoard.inHand(game, a, "Mountain"));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(mountain, a);

        assertFalse(verdict.canPlay());
        assertTrue(verdict.timing());
        assertEquals(List.of(PlayabilityCheck.LAND_PLAY_LIMIT), checks(verdict));
    }

    /** Exploration's second land play makes it playable again. */
    @Test
    void explorationRaisesTheLandPlayLimit() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Exploration");
        game.getAction().checkStaticAbilities();
        a.playLand(TestBoard.inHand(game, a, "Forest"), null);
        SpellAbility mountain = landPlayOf(TestBoard.inHand(game, a, "Mountain"));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(mountain, a);

        assertEquals(2, a.getMaxLandPlays(), "the board under test");
        assertTrue(verdict.canPlay());
        assertEquals(List.of(), verdict.failedChecks());
    }

    /** Rule 305.3: a land off its owner's turn fails whose turn, and with it timing. */
    @Test
    void aLandOffTurnFailsWhoseTurnAndCastingSpeed() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player b = game.getPlayers().get(1);
        SpellAbility island = landPlayOf(TestBoard.inHand(game, b, "Island"));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(island, b);

        assertFalse(verdict.canPlay());
        assertFalse(verdict.timing());
        assertEquals(List.of(PlayabilityCheck.CASTING_SPEED, PlayabilityCheck.WHOSE_TURN),
                checks(verdict));
    }

    /** Aggressive Mining's "You can't play lands" is a static prohibition naming its static. */
    @Test
    void aStaticThatForbidsLandPlaysNamesItsStatic() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card mining = TestBoard.inPlay(game, a, "Aggressive Mining");
        SpellAbility forest = landPlayOf(TestBoard.inHand(game, a, "Forest"));
        ProvenanceKey cantPlayLand =
                ProvenanceResolver.of(staticOf(mining, StaticAbilityMode.CantPlayLand));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(forest, a);

        assertFalse(verdict.canPlay());
        assertEquals(List.of(new PlayabilityVerdict.FailedCheck(
                PlayabilityCheck.STATIC_PROHIBITION, cantPlayLand)), verdict.failedChecks());
        assertEquals(cantPlayLand, verdict.prohibitingStatic());
    }

    /** A land on the battlefield fails zone. */
    @Test
    void aLandOnTheBattlefieldFailsZone() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility forest = landPlayOf(TestBoard.inPlay(game, a, "Forest"));

        PlayabilityVerdict verdict = PlayabilityVerdict.of(forest, a);

        assertFalse(verdict.canPlay());
        assertEquals(List.of(PlayabilityCheck.ZONE), checks(verdict));
    }

    /** A land play's record carries no mana, no target and {@code land_play = true}. */
    @Test
    void aLandPlaysCandidateCarriesLandPlayAndNoCost() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility forest = landPlayOf(TestBoard.inHand(game, a, "Forest"));

        DecisionCandidate read = DecisionCandidate.read(forest, a).orElseThrow();

        assertTrue(read.landPlay());
        assertNull(read.manaCost());
        assertEquals(List.of(), read.legalTargets());
        assertTrue(read.payloadJson().contains("\"land_play\":true"), read.payloadJson());
    }
}
