package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.GameRecording;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.SnapshotBuilder;
import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The random seat's decision points, on scripted boards (spec Story 3
 * scenarios 1–5 and 15–18).
 *
 * <p>Each board is a real two-player game with real cards in real zones,
 * built the way {@code CopiedTokenProvenanceTest} builds its board, and the
 * seat is the controller {@link RandomSeatLobbyPlayer} installs when the game
 * creates its players — the installation under test, not a stand-in for it.
 * {@code P} is one throughout, so every decision point draws and the
 * question is what the draw does; the gate itself is a coin, and the
 * share-of-matches test reads it off the match-level draw instead.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatControllerTest {

    @TempDir
    Path tempDir;

    /** A game whose seat A is the random seat, in {@code phase} on A's turn. */
    private static Game gameWithRandomSeatA(PhaseType phase, double probability, long seed) {
        return TestBoard.twoPlayerGame(phase,
                new RandomSeatLobbyPlayer("a", probability, new Random(seed)),
                new LobbyPlayerAi("b", null));
    }

    private static RandomSeatController seatOf(Game game) {
        RandomSeatController seat = GamePlayer.randomSeatOf(game);
        assertNotNull(seat, "the lobby player installs the controller");
        return seat;
    }

    /**
     * Seat A holding Pacifism, whose one legal target is A's own creature.
     *
     * <p>The AI never enchants its own creature with it, so on this board a
     * Pacifism chosen is a random draw's choice and anything else is the AI's.
     */
    private static Card pacifismWithOnlyAnOwnTarget(Game game, Player a) {
        TestBoard.inPlay(game, a, "Plains");
        TestBoard.inPlay(game, a, "Plains");
        TestBoard.inPlay(game, a, "Grizzly Bears");
        return TestBoard.inHand(game, a, "Pacifism");
    }

    /** Whether {@code chosen} plays a card named {@code name}. */
    private static boolean plays(List<SpellAbility> chosen, String name) {
        return chosen != null && chosen.stream()
                .anyMatch(sa -> name.equals(sa.getHostCard().getName()));
    }

    // ── the play decision ───────────────────────────────────────────────

    /** Scenario 3: a seat whose only playable abilities are mana abilities passes. */
    @Test
    void aSeatWithOnlyManaAbilitiesPlayablePasses() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inPlay(game, a, "Forest");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        assertNull(seat.chooseSpellAbilityToPlay(), "nothing but mana to play: pass");
    }

    /**
     * Scenario 3 and the one-option edge case: the single playable spell is
     * taken, though it is one the AI would pass on.
     */
    @Test
    void aSingleLegalOptionIsTaken() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        Card pacifism = pacifismWithOnlyAnOwnTarget(game, a);
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();

        assertNotNull(chosen, "a random draw never passes while anything is playable");
        assertEquals(1, chosen.size());
        assertEquals(pacifism, chosen.get(0).getHostCard());
    }

    /**
     * Scenario 15's first half, end to end: the drawn play is cast with the
     * AI paying its mana, and lands on the stack.
     */
    @Test
    void aRandomlyDrawnSpellIsCastOntoTheStack() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inHand(game, a, "Grizzly Bears");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        SpellAbility bears = seat.chooseSpellAbilityToPlay().get(0);
        assertTrue(seat.playChosenSpellAbility(bears));

        assertEquals(1, game.getStack().size(), "the spell is on the stack");
        assertTrue(a.getCardsIn(ZoneType.Hand).isEmpty(), "and no longer in hand");
    }

    /**
     * Scenario 18: holding a playable land with the drop unused, the seat
     * makes no draw and the AI decides; once the drop is used, it draws.
     */
    @Test
    void landFirstThenTheDraw() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        pacifismWithOnlyAnOwnTarget(game, a);
        Card land = TestBoard.inHand(game, a, "Plains");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        // A draw never picks a land, so a land played is the AI deciding.
        List<SpellAbility> first = seat.chooseSpellAbilityToPlay();
        assertNotNull(first, "no draw while a land drop is open: the AI plays the land");
        assertTrue(first.get(0).isLandAbility(), first.toString());
        assertEquals(land, first.get(0).getHostCard());

        a.playLand(land, null);
        assertEquals(1, a.getLandsPlayedThisTurn());

        List<SpellAbility> second = seat.chooseSpellAbilityToPlay();
        assertTrue(plays(second, "Pacifism"), "the drop is used: the draw is made, " + second);
    }

    /** Scenario 2's other branch: a draw that fails leaves the decision to the AI. */
    @Test
    void aFailedDrawLeavesThePlayToTheAi() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 0.0, 1L);
        Player a = game.getPlayers().get(0);
        pacifismWithOnlyAnOwnTarget(game, a);
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        for (int i = 0; i < 20; i++) {
            List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
            assertFalse(plays(chosen, "Pacifism"), "P = 0 never draws: " + chosen);
        }
    }

    /**
     * Scenario 15's last clause: a randomly drawn play whose draws keep
     * failing is abandoned and the play draw repeats over the rest. Fiery
     * Intervention is playable by every check the candidate filter makes —
     * the charm itself targets nothing — but on a board with no creature and
     * no artifact neither of its modes has a target, so every mode draw
     * fails; Shock always has one. Whichever the first draw picks, Shock is
     * what reaches the stack.
     */
    @Test
    void aPlayThatCannotBeCompletedIsAbandonedForAnother() {
        for (long seed = 1; seed <= 6; seed++) {
            Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, seed);
            Player a = game.getPlayers().get(0);
            for (int i = 0; i < 5; i++) {
                TestBoard.inPlay(game, a, "Mountain");
            }
            TestBoard.inHand(game, a, "Fiery Intervention");
            TestBoard.inHand(game, a, "Shock");
            RandomSeatController seat = seatOf(game);
            seat.attach(game, null);

            List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
            assertNotNull(chosen, "both are playable");
            assertTrue(seat.playChosenSpellAbility(chosen.get(0)), "seed " + seed);
            assertEquals(1, game.getStack().size());
            assertEquals("Shock", game.getStack().peekAbility().getHostCard().getName(),
                    "seed " + seed + " first drew " + chosen.get(0).getHostCard());
        }
    }

    /**
     * A charm none of whose modes can be chosen is no candidate at all, so
     * the seat passes rather than drawing a play it would have to abandon.
     */
    @Test
    void aCharmWithNoChoosableModeIsNotACandidate() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            TestBoard.inPlay(game, a, "Mountain");
        }
        TestBoard.inHand(game, a, "Fiery Intervention");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        assertNull(seat.chooseSpellAbilityToPlay(), "no creature, no artifact: nothing to draw, pass");
    }

    /**
     * Scenario 15: a randomly drawn charm's modes are drawn, not the AI's —
     * Fiery Confluence chooses three with repeats, and across seeds the
     * chosen lists differ, repeats included.
     */
    @Test
    void aRandomlyDrawnCharmCastsWithDrawnModes() {
        java.util.Set<List<Integer>> mixes = new java.util.HashSet<>();
        boolean repeated = false;
        for (long seed = 1; seed <= 30; seed++) {
            Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, seed);
            Player a = game.getPlayers().get(0);
            for (int i = 0; i < 4; i++) {
                TestBoard.inPlay(game, a, "Mountain");
            }
            TestBoard.inHand(game, a, "Fiery Confluence");
            RandomSeatController seat = seatOf(game);
            seat.attach(game, null);

            SpellAbility charm = seat.chooseSpellAbilityToPlay().get(0);
            assertTrue(seat.playChosenSpellAbility(charm), "seed " + seed);
            SpellAbility onStack = game.getStack().peekAbility();
            List<forge.game.spellability.AbilitySub> choices =
                    onStack.getAdditionalAbilityList("Choices");
            List<Integer> mix = new ArrayList<>();
            for (var mode : onStack.getChosenList()) {
                mix.add(choices.indexOf(mode));
            }
            assertEquals(3, mix.size(), "CharmNum 3: " + mix);
            repeated |= new java.util.HashSet<>(mix).size() < mix.size();
            mixes.add(mix);
        }
        assertTrue(mixes.size() > 3, "the modes are drawn: " + mixes);
        assertTrue(repeated, "a charm that may repeat modes sometimes does");
    }

    /** Scenario 4 by construction: mulligans, mana and land choice are never overridden. */
    @Test
    void landManaAndMulliganDecisionsAreNotOverridden() throws NoSuchMethodException {
        for (String method : List.of(
                "mulliganKeepHand", "payManaCost", "chooseManaFromPool", "tuckCardsViaMulligan")) {
            boolean overridden = false;
            for (var declared : RandomSeatController.class.getDeclaredMethods()) {
                overridden |= declared.getName().equals(method);
            }
            assertFalse(overridden, method + " must stay the AI's (FR-023)");
        }
    }

    /**
     * T063: a fork of the game does not act at random. {@code GameCopier}
     * clones the lobby player, so the fork's player gets a controller of this
     * class too; never attached, it is the ordinary AI.
     */
    @Test
    void aForkOfTheGameDoesNotActAtRandom() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        pacifismWithOnlyAnOwnTarget(game, a);
        seatOf(game).attach(game, null);

        Game fork = new GameCopier(game).makeCopy();
        RandomSeatController forked = GamePlayer.randomSeatOf(fork);
        assertNotNull(forked, "the fork keeps the lobby-player subclass");
        assertFalse(forked.isLive());

        List<SpellAbility> chosen = forked.chooseSpellAbilityToPlay();
        assertFalse(plays(chosen, "Pacifism"), "a fork's seat defers to the AI: " + chosen);
    }

    // ── attacks and blocks ──────────────────────────────────────────────

    /**
     * Scenario 2: with nothing restricting attacks, each creature attacks in
     * close to half the draws. The seat writes no {@code attackers} record of
     * its own: the engine reports legality from the turn-based declaration,
     * before any controller declares, and that report covers this seat as it
     * covers every other.
     */
    @Test
    void eachCreatureAttacksInAboutHalfTheDrawsAndTheSeatWritesNoLegalityRecord() {
        Game game = gameWithRandomSeatA(PhaseType.COMBAT_DECLARE_ATTACKERS, 1.0, 7L);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        List<Card> creatures = List.of(
                TestBoard.inPlay(game, a, "Grizzly Bears"),
                TestBoard.inPlay(game, a, "Runeclaw Bear"),
                TestBoard.inPlay(game, a, "Hill Giant"),
                TestBoard.inPlay(game, a, "Grizzly Bears"));
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        RandomSeatController seat = seatOf(game);
        int draws = 400;
        int[] attacks = new int[creatures.size()];
        try (GameRecording recording = new GameRecording(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L,
                SnapshotBuilder.playerId(a))) {
            seat.attach(game, recording.playability());
            for (int draw = 0; draw < draws; draw++) {
                Combat combat = new Combat(a);
                game.getPhaseHandler().setCombat(combat);
                seat.declareAttackers(a, combat);
                for (int i = 0; i < creatures.size(); i++) {
                    if (combat.isAttacking(creatures.get(i))) {
                        attacks[i]++;
                        assertEquals(b, combat.getDefenderByAttacker(creatures.get(i)));
                    }
                }
            }
        } finally {
            writer.close();
        }

        for (int i = 0; i < creatures.size(); i++) {
            double share = attacks[i] / (double) draws;
            assertTrue(share > 0.4 && share < 0.6,
                    creatures.get(i) + " attacked in " + share + " of the draws");
        }
        // Nothing at all: the seat's declarations write no record of any kind.
        assertFalse(Files.exists(path),
                "the engine's declaration hook records attacks, not the seat");
    }

    /**
     * Scenario 17: the random seat draws both a block and no block, and the
     * blocks it declares are its own creatures'. As with attacks, the seat
     * writes no {@code blockers} record of its own.
     */
    @Test
    void aBlockDeclarationDrawsBothWaysAndTheSeatWritesNoLegalityRecord() {
        Game game = gameWithRandomSeatA(PhaseType.COMBAT_DECLARE_BLOCKERS, 1.0, 3L);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        // B's turn: B attacks A with one creature, A has two it could block with.
        game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DECLARE_BLOCKERS, b);
        Card attacker = TestBoard.inPlay(game, b, "Hill Giant");
        List<Card> blockers = List.of(
                TestBoard.inPlay(game, a, "Grizzly Bears"),
                TestBoard.inPlay(game, a, "Runeclaw Bear"));
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        RandomSeatController seat = seatOf(game);
        boolean blockedOnce = false;
        boolean heldBackOnce = false;
        try (GameRecording recording = new GameRecording(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L,
                SnapshotBuilder.playerId(a))) {
            seat.attach(game, recording.playability());
            for (int draw = 0; draw < 100; draw++) {
                Combat combat = new Combat(b);
                combat.addAttacker(attacker, a);
                game.getPhaseHandler().setCombat(combat);
                seat.declareBlockers(a, combat);
                List<Card> declared = new ArrayList<>(combat.getBlockers(attacker));
                blockedOnce |= !declared.isEmpty();
                heldBackOnce |= declared.isEmpty();
                for (Card blocker : declared) {
                    assertTrue(blockers.contains(blocker), "only the seat's creatures block");
                }
            }
        } finally {
            writer.close();
        }

        assertTrue(blockedOnce && heldBackOnce, "no block and a block are both drawn");
        assertFalse(Files.exists(path),
                "the engine's declaration hook records blocks, not the seat");
    }

    // ── the match-level draw ────────────────────────────────────────────

    /** Scenario 1: one match in eight seats a random player, A or B alike. */
    @Test
    void closeToOneMatchInEightSeatsARandomPlayerOnEitherSide() {
        CollectionCaps caps = new CollectionCaps(
                1, 0.1, 2, 2, List.of(), 0.125, 0.5);
        Random random = new Random(11L);
        int draws = 40_000;
        int a = 0;
        int b = 0;
        for (int i = 0; i < draws; i++) {
            Optional<Side> seat = GamePlayer.drawRandomSeat(caps, random);
            if (seat.equals(Optional.of(Side.A))) {
                a++;
            } else if (seat.equals(Optional.of(Side.B))) {
                b++;
            }
        }
        double share = (a + b) / (double) draws;
        assertTrue(share > 0.11 && share < 0.14, "share " + share);
        assertTrue(a > 0.4 * (a + b) && a < 0.6 * (a + b), "seat A " + a + " of " + (a + b));
    }

    @Test
    void aShareOfZeroSeatsNobody() {
        Random random = new Random(1L);
        for (int i = 0; i < 1000; i++) {
            assertEquals(Optional.empty(),
                    GamePlayer.drawRandomSeat(CollectionCaps.defaults(), random));
        }
    }
}
