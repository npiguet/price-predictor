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
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
    void aPermanentsActivatedAbilityIsForcedOnTheBattlefieldAndOnlyItsSpellInTheHand() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card inPlay = TestBoard.inPlay(game, you, "Prodigal Sorcerer");
        Card inHand = TestBoard.inHand(game, you, "Prodigal Sorcerer");

        List<SpellAbility> onBattlefield = InterventionScheduler.forcedFrom(inPlay, ZoneType.Battlefield);
        assertEquals(1, onBattlefield.size(), onBattlefield.toString());
        assertTrue(onBattlefield.get(0).isActivatedAbility());
        List<SpellAbility> fromHand = InterventionScheduler.forcedFrom(inHand, ZoneType.Hand);
        assertEquals(1, fromHand.size(), fromHand.toString());
        assertTrue(fromHand.get(0).isSpell(), "its tap ability is not activated with the card in hand");
    }

    @Test
    void aCyclingAbilityIsForcedFromTheHand() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card tusker = TestBoard.inHand(game, you, "Krosan Tusker");

        List<SpellAbility> fromHand = InterventionScheduler.forcedFrom(tusker, ZoneType.Hand);
        assertTrue(fromHand.stream().anyMatch(SpellAbility::isSpell), fromHand.toString());
        assertTrue(fromHand.stream().anyMatch(SpellAbility::isCycling), fromHand.toString());
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
    // ── a steered game's first pass (FR-058e) ───────────────────────────────

    /**
     * In a steered game the variant cards each player holds are forced first,
     * the active player's then the other's, before FR-057's pass and the
     * general pass, which then skip a line the first pass forced.
     */
    @Test
    void aSteeredGameForcesTheVariantCardsFirst(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);

        assertEquals(List.of("Volcanic Hammer@0", "Searing Spear@1", "Lightning Bolt@0"),
                board.forced(true, 10));
    }

    /** Without a steered deck the variants are lines like any other. */
    @Test
    void anUnsteeredGameHasNoVariantPass(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);

        assertEquals(List.of("Lightning Bolt@0", "Volcanic Hammer@0"), board.forced(false, 10));
    }

    /** The first pass spends the one budget the other passes share. */
    @Test
    void theVariantPassSpendsTheSameBudget(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);

        assertEquals(List.of("Volcanic Hammer@0"), board.forced(true, 1));
    }

    /**
     * Two players at MAIN1 of the first one's turn: the first holds Lightning
     * Bolt and the variant Volcanic Hammer, the second the variant Searing
     * Spear. Both names are registered as variants here and named by no other
     * test, since the registry is JVM-wide.
     */
    private static Board variantBoard(Path dir) throws IOException {
        Files.writeString(dir.resolve("volcanic_hammer.txt"), "Name:Volcanic Hammer\n");
        Files.writeString(dir.resolve("searing_spear.txt"), "Name:Searing Spear\n");
        VariantRegistry.registerAll(dir);
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inHand(game, active, "Lightning Bolt");
        TestBoard.inHand(game, active, "Volcanic Hammer");
        TestBoard.inHand(game, other, "Searing Spear");
        return new Board(game);
    }

    private record Board(Game game) {

        /** What a scheduler forces at one boundary, as {@code card@holder index}, under {@code budget}. */
        List<String> forced(boolean steered, int budget) {
            List<String> forced = new ArrayList<>();
            InterventionScheduler.Forcing recording = new InterventionScheduler.Forcing() {
                @Override
                public boolean mayIntervene(int realResolutionId) {
                    return forced.size() < budget;
                }

                @Override
                public boolean intervene(SpellAbility ability, int realResolutionId, Player holder) {
                    forced.add(ability.getHostCard().getName() + "@" + game.getPlayers().indexOf(holder));
                    return true;
                }
            };
            new InterventionScheduler(game, recording, null, steered).collect();
            return forced.stream()
                    .filter(line -> line.startsWith("Lightning Bolt") || line.startsWith("Volcanic")
                            || line.startsWith("Searing"))
                    .toList();
        }
    }
}
