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
        seen.add(InterventionScheduler.lineKey(bolt));
        assertFalse(InterventionScheduler.worthIntervening(bolt, seen));
        assertFalse(InterventionScheduler.worthIntervening(
                TestCards.build("Lightning Bolt").getFirstSpellAbility(), seen),
                "another copy of the card is the same line");
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

        assertEquals(List.of("Volcanic Hammer@0"), board.forced(true, 10, 1),
                "one line a boundary");
        assertEquals(List.of("Volcanic Hammer@0", "Searing Spear@1", "Lightning Bolt@0"),
                board.forced(true, 10, 10));
    }

    /** Without a steered deck the variants are lines like any other. */
    @Test
    void anUnsteeredGameHasNoVariantPass(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);

        assertEquals(List.of("Lightning Bolt@0", "Volcanic Hammer@0"), board.forced(false, 10, 10));
    }

    /** The first pass spends the one budget the other passes share. */
    @Test
    void theVariantPassSpendsTheSameBudget(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);

        assertEquals(List.of("Volcanic Hammer@0"), board.forced(true, 1, 10));
    }

    /**
     * A boundary that does not draw a line forces nothing, so the budget is
     * not all spent where the game starts; the gate is drawn once a boundary.
     */
    @Test
    void aBoundaryThatDrawsNoLineForcesNothing(@TempDir Path dir) throws IOException {
        Board board = variantBoard(dir);
        List<String> forced = new ArrayList<>();
        int[] draws = {0};
        InterventionScheduler scheduler = new InterventionScheduler(
                board.game(), board.recording(forced, 10), null, false, (turn, phase) -> draws[0]++ == 1);

        scheduler.collect();
        assertEquals(List.of(), forced);
        scheduler.collect();
        assertEquals(1, forced.size(), forced.toString());
        assertEquals(2, draws[0]);
    }

    /** The worker's gate draws the same turns for one seed, and from the game's seed alone. */
    @Test
    void theWorkersGateDependsOnTheGamesSeedAlone() {
        for (long seed = 0; seed < 50; seed++) {
            InterventionScheduler.Gate gate = InterventionScheduler.spreadOverTheGame(seed, 2);
            InterventionScheduler.Gate again = InterventionScheduler.spreadOverTheGame(seed, 2);
            for (int turn = 1; turn <= 40; turn++) {
                boolean open = gate.open(turn, PhaseType.END_OF_TURN);
                assertEquals(open, again.open(turn, PhaseType.END_OF_TURN), "seed " + seed + " turn " + turn);
                if (open) {
                    gate.forced();
                    again.forced();
                }
            }
        }
    }

    /**
     * The budget is spread over a typical game: of games forcing at every
     * turn the gate opens, about half have drawn their first intervention
     * by the typical game's middle and all their budget by its end, and a
     * game shorter than typical may end with some unspent.
     */
    @Test
    void theWorkersGateSpreadsTheBudgetOverATypicalGame() {
        int typical = InterventionScheduler.TYPICAL_GAME_TURNS;
        int games = 2000;
        int spentByTheEnd = 0;
        int firstByTheMiddle = 0;
        int spentByAFifth = 0;
        for (long seed = 0; seed < games; seed++) {
            InterventionScheduler.Gate gate = InterventionScheduler.spreadOverTheGame(seed, 2);
            int forced = 0;
            for (int turn = 1; turn <= typical; turn++) {
                if (gate.open(turn, PhaseType.END_OF_TURN)) {
                    gate.forced();
                    forced++;
                    if (forced == 1 && turn <= typical / 2) {
                        firstByTheMiddle++;
                    }
                }
                if (turn == typical / 5 && forced == 2) {
                    spentByAFifth++;
                }
            }
            spentByTheEnd += forced == 2 ? 1 : 0;
        }
        assertEquals(games, spentByTheEnd, "the whole budget by the typical game's last turn");
        assertTrue(firstByTheMiddle > games * 0.3 && firstByTheMiddle < games * 0.85, firstByTheMiddle + "");
        assertTrue(spentByAFifth < games * 0.1, spentByAFifth + " spent within the first fifth");
    }

    /**
     * The step an intervention opens at is drawn with it: over many games
     * the first opening falls at every step that gives priority, and never
     * before the turn's upkeep.
     */
    @Test
    void anInterventionOpensAtAStepDrawnWithIt() {
        Set<PhaseType> opened = new HashSet<>();
        for (long seed = 0; seed < 500; seed++) {
            InterventionScheduler.Gate gate = InterventionScheduler.spreadOverTheGame(seed, 1);
            turns:
            for (int turn = 1; turn <= 40; turn++) {
                for (PhaseType phase : PhaseType.values()) {
                    if (gate.open(turn, phase)) {
                        opened.add(phase);
                        break turns;
                    }
                }
            }
        }
        assertEquals(Set.copyOf(InterventionScheduler.OPENING_STEPS), opened);
    }

    /** An intervention drawn and not forced is carried to the next boundary, never drawn twice. */
    @Test
    void aDrawnInterventionIsCarriedUntilALineIsForced() {
        InterventionScheduler.Gate gate = InterventionScheduler.spreadOverTheGame(3L, 1);
        int turn = 1;
        while (!gate.open(turn, PhaseType.END_OF_TURN)) {
            turn++;
        }
        assertTrue(gate.open(turn + 1, PhaseType.UPKEEP), "carried to the next turn");
        assertTrue(gate.open(turn + 5, PhaseType.UPKEEP), "and on");
        gate.forced();
        for (int later = turn + 5; later < turn + 40; later++) {
            assertFalse(gate.open(later, PhaseType.END_OF_TURN), "a budget of one, spent");
        }
    }

    /**
     * A line its timing forbids at this boundary is passed over and not
     * marked: a sorcery at the opponent's turn or outside a main phase, any
     * line at the cleanup step; an instant at the opponent's upkeep is forced.
     */
    @Test
    void aLineIsForcedOnlyWhereItsTimingLetsItsHolderPlayIt() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        SpellAbility lavaAxe = TestBoard.inHand(game, active, "Lava Axe").getFirstSpellAbility();
        SpellAbility bolt = TestBoard.inHand(game, other, "Lightning Bolt").getFirstSpellAbility();

        assertFalse(InterventionScheduler.playableNow(lavaAxe, active), "a sorcery in the upkeep");
        assertTrue(InterventionScheduler.playableNow(bolt, other), "an instant in the opponent's upkeep");
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, active, 2);
        assertTrue(InterventionScheduler.playableNow(lavaAxe, active), "a sorcery in its holder's main phase");
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, other, 2);
        assertFalse(InterventionScheduler.playableNow(lavaAxe, active), "a sorcery in the opponent's main phase");
        game.getPhaseHandler().devModeSet(PhaseType.CLEANUP, other, 2);
        assertFalse(InterventionScheduler.playableNow(bolt, other), "no priority in the cleanup step");
    }

    /**
     * Karn Liberated's restart resets the JVM-wide trigger id counter
     * ({@code RestartGameEffect} calls {@code Trigger.resetIDs()}), which a
     * fork would do under the live game: the line is never forced.
     */
    @Test
    void aLineThatRestartsTheGameIsNeverForced() {
        Card karn = TestCards.build("Karn Liberated");
        List<SpellAbility> restarts = karn.getAllSpellAbilities().stream()
                .filter(InterventionScheduler::restartsTheGame).toList();

        assertEquals(1, restarts.size(), karn.getAllSpellAbilities().toString());
        assertFalse(InterventionScheduler.worthIntervening(restarts.get(0), new HashSet<>()));
    }

    /**
     * A ninjutsu line reads the attacker its cost returned, which a forced
     * line never pays: it is passed over as a game-restart line is, so the
     * boundary forces the next line and the gate's planned intervention goes
     * to it; the refusal is counted once a game, however often it is met.
     */
    @Test
    void aLineThatReadsAnUnpaidCostTakesNeitherTheBoundaryNorAPlannedIntervention() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        Card ninja = TestBoard.inHand(game, active, "Ninja of the Deep Hours");
        TestBoard.inHand(game, active, "Lightning Bolt");
        SpellAbility ninjutsu = ninja.getSpellAbilities().stream()
                .filter(SpellAbility::isNinjutsu).findFirst().orElseThrow();
        ninjutsu.setActivatingPlayer(active);
        assertTrue(InterventionScheduler.playableNow(ninjutsu, active), "ninjutsu is not playable at the upkeep");
        String refusal = "intervention " + InterventionFork.READS_AN_UNPAID_COST;
        long refused = ForkRefusals.counts().getOrDefault(refusal, 0L);
        List<String> forced = new ArrayList<>();
        int[] planned = {0};
        InterventionScheduler.Gate gate = new InterventionScheduler.Gate() {
            @Override
            public boolean open(int turn, PhaseType phase) {
                return true;
            }

            @Override
            public void forced() {
                planned[0]++;
            }
        };
        InterventionScheduler scheduler = new InterventionScheduler(
                game, new Board(game).recording(forced, 10), null, false, gate);

        scheduler.collect();
        assertEquals(List.of("Lightning Bolt@0"), forced, "the boundary went to the ninjutsu line");
        assertEquals(1, planned[0], "planned interventions spent");
        scheduler.collect();
        assertEquals(List.of("Lightning Bolt@0"), forced, "nothing else to force");
        assertEquals(1, planned[0], "planned interventions spent");
        assertEquals(refused + 1, ForkRefusals.counts().getOrDefault(refusal, 0L), ForkRefusals.counts().toString());
    }

    /**
     * Two players at MAIN1 of the first one's turn: the first holds Lightning
     * Bolt and the variant Volcanic Hammer, the second the variant Searing
     * Spear. Both names are registered as variants here and forgotten after
     * each test ({@link #forgetTheVariants}), since the registry is JVM-wide.
     */
    /**
     * The two variant names put back after each test: they are real cards,
     * and left registered they would key every later test's Volcanic Hammer
     * or Searing Spear under {@code variant-scripts/}.
     */
    @org.junit.jupiter.api.AfterEach
    void forgetTheVariants() {
        VariantRegistry.forget("Volcanic Hammer", "Searing Spear");
    }

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

        /**
         * What a scheduler forces over {@code boundaries} boundaries, each
         * drawing a line, as {@code card@holder index}, under {@code budget}.
         */
        List<String> forced(boolean steered, int budget, int boundaries) {
            List<String> forced = new ArrayList<>();
            InterventionScheduler scheduler = new InterventionScheduler(
                    game, recording(forced, budget), null, steered, InterventionScheduler.EVERY_BOUNDARY);
            for (int boundary = 0; boundary < boundaries; boundary++) {
                scheduler.collect();
            }
            return forced.stream()
                    .filter(line -> line.startsWith("Lightning Bolt") || line.startsWith("Volcanic")
                            || line.startsWith("Searing"))
                    .toList();
        }

        /** A stand-in for the forks that writes down what it is asked to force. */
        InterventionScheduler.Forcing recording(List<String> forced, int budget) {
            return new InterventionScheduler.Forcing() {
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
        }
    }
}
