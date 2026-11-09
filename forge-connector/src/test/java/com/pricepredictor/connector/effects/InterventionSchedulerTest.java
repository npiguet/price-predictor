package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which lines an intervention may force.
 *
 * <p>Checked here rather than against a game's records: a protection fork plays
 * its turn on and records the lands it taps as a live game does, so a basic
 * land's mana record carrying {@code interventional = true} is no forced one.
 */
@ExtendWith(ForgeExtension.class)
class InterventionSchedulerTest {

    @Test
    void neitherALandNorItsManaAbilityIsForced() {
        Card forest = TestCards.build("Forest");
        List<SpellAbility> abilities = List.copyOf(forest.getAllSpellAbilities());

        assertTrue(abilities.stream().anyMatch(SpellAbility::isLandAbility), "no land ability to test");
        assertTrue(abilities.stream().anyMatch(SpellAbility::isManaAbility), "no mana ability to test");
        for (SpellAbility ability : abilities) {
            assertFalse(InterventionScheduler.worthIntervening(ability, new HashSet<>()),
                    "would force " + ability);
        }
    }

    @Test
    void aSpellIsForcedOncePerLine() {
        SpellAbility bolt = TestCards.build("Lightning Bolt").getFirstSpellAbility();
        Set<String> seen = new HashSet<>();

        assertTrue(InterventionScheduler.worthIntervening(bolt, seen));
        seen.add(AbilityLines.key(bolt));
        assertFalse(InterventionScheduler.worthIntervening(bolt, seen));
    }

    @Test
    void theHandIsStillForcedFirst() {
        assertEquals(ZoneType.Hand, InterventionScheduler.FORCED_ZONES.get(0));
    }

    @Test
    void aPlaneswalkersLoyaltyAbilitiesAreForcedOnTheBattlefieldAndItsSpellIsNot() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card ajani = TestBoard.inPlay(game, you, "Ajani Goldmane");

        List<SpellAbility> lines = InterventionScheduler.forcedFrom(ajani, ZoneType.Battlefield);

        assertEquals(3, lines.size(), lines.toString());
        assertTrue(lines.stream().allMatch(SpellAbility::isPwAbility), lines.toString());
    }

    @Test
    void aPermanentsActivatedAbilityIsForcedOnTheBattlefieldAndAllItsLinesInTheHand() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card inPlay = TestBoard.inPlay(game, you, "Prodigal Sorcerer");
        Card inHand = TestBoard.inHand(game, you, "Prodigal Sorcerer");

        List<SpellAbility> onBattlefield = InterventionScheduler.forcedFrom(inPlay, ZoneType.Battlefield);
        assertEquals(1, onBattlefield.size(), onBattlefield.toString());
        assertTrue(onBattlefield.get(0).isActivatedAbility());
        assertTrue(InterventionScheduler.forcedFrom(inHand, ZoneType.Hand).stream().anyMatch(SpellAbility::isSpell));
    }

    @Test
    void aGraveyardAbilityIsForcedFromTheGraveyardAndNowhereElse() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card inGraveyard = TestBoard.inGraveyard(game, you, "Reassembling Skeleton");
        Card inPlay = TestBoard.inPlay(game, you, "Reassembling Skeleton");

        List<SpellAbility> fromGraveyard = InterventionScheduler.forcedFrom(inGraveyard, ZoneType.Graveyard);
        assertEquals(1, fromGraveyard.size(), fromGraveyard.toString());
        assertFalse(fromGraveyard.get(0).isSpell(), "the creature spell is not cast from the graveyard");
        assertTrue(InterventionScheduler.forcedFrom(inPlay, ZoneType.Battlefield).isEmpty());
    }
}
