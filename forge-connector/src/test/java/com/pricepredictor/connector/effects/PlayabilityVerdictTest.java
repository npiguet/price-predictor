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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The rules' verdict on a candidate, on boards where the AI's answer and the
 * rules' answer disagree — the only boards where a test can tell them apart.
 */
@ExtendWith(ForgeExtension.class)
class PlayabilityVerdictTest {

    private static SpellAbility abilityOn(Card host, String script) {
        SpellAbility ability = AbilityFactory.getAbility(script, host);
        ability.setActivatingPlayer(host.getController());
        return ability;
    }

    /** No target has been chosen, which is what the AI leaves when it stops early; the rules still offer one. */
    @Test
    void aTargetTheRulesOfferCountsWhetherOrNotOneWasChosen() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility pump = abilityOn(bears, "AB$ Pump | Cost$ 0 | ValidTgts$ Creature | NumAtt$ +1");

        assertTrue(pump.getTargets().isEmpty(), "the board under test has no chosen target");
        assertTrue(PlayabilityVerdict.of(pump).hasLegalTarget());
    }

    /** And a target the rules do not offer is absent however the ability was left. */
    @Test
    void aTargetTheRulesDoNotOfferIsAbsent() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility destroy = abilityOn(bears, "AB$ Destroy | Cost$ 0 | ValidTgts$ Artifact");

        assertFalse(PlayabilityVerdict.of(destroy).hasLegalTarget());
    }

    /** Fiery Intervention: 5 damage to target creature, or destroy target artifact. */
    private static SpellAbility fieryIntervention(Game game, Player owner) {
        Card card = TestBoard.inHand(game, owner, "Fiery Intervention");
        SpellAbility charm = card.getFirstSpellAbility();
        charm.setActivatingPlayer(owner);
        return charm;
    }

    /** One choosable mode is all a choose-one charm asks for, and none is too few. */
    @Test
    void aCharmIsPlayableWhenEnoughOfItsModesCanBeChosen() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        SpellAbility charm = fieryIntervention(game, a);

        assertFalse(PlayabilityVerdict.of(charm).hasLegalTarget(),
                "no creature and no artifact: neither mode can be chosen");
        TestBoard.inPlay(game, a, "Grizzly Bears");
        assertTrue(PlayabilityVerdict.of(charm).hasLegalTarget(),
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

        assertTrue(PlayabilityVerdict.of(charm).hasLegalTarget());
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
        assertTrue(PlayabilityVerdict.of(minusOne).affordable());
    }

    @Test
    void aCandidateWithNoActivatingPlayerGetsNoVerdict() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        SpellAbility pump = AbilityFactory.getAbility("AB$ Pump | Cost$ 0 | NumAtt$ +1", bears);

        assertFalse(PlayabilityVerdict.of(pump).canPlay());
        assertFalse(PlayabilityVerdict.of(null).affordable());
    }
}
