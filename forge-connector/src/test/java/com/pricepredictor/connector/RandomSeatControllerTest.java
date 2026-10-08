package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.SnapshotBuilder;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.ai.simulation.GameCopier;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

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
        Deck deck = new Deck();
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(deck).setPlayer(
                        new RandomSeatLobbyPlayer("a", probability, new Random(seed))),
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(players, rules, new Match(rules, players, "RandomSeatControllerTest"));
        game.getPhaseHandler().devModeSet(phase, game.getPlayers().get(0));
        return game;
    }

    private static RandomSeatController seatOf(Game game) {
        RandomSeatController seat = GamePlayer.randomSeatOf(game);
        assertNotNull(seat, "the lobby player installs the controller");
        return seat;
    }

    private static Card inPlay(Game game, Player owner, String name) {
        Card card = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(name), owner, game);
        game.getAction().moveToPlay(card, owner, null, null);
        card.setSickness(false);
        return card;
    }

    private static Card inHand(Game game, Player owner, String name) {
        Card card = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(name), owner, game);
        return game.getAction().moveTo(ZoneType.Hand, card, null, null);
    }

    private List<String> readShard(Path path) throws IOException {
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }

    // ── the play decision ───────────────────────────────────────────────

    /** Scenario 3: a seat whose only playable abilities are mana abilities passes. */
    @Test
    void aSeatWithOnlyManaAbilitiesPlayablePasses() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        assertNull(seat.chooseSpellAbilityToPlay(), "nothing but mana to play: pass");
    }

    /** Scenario 3 and the one-option edge case: the single playable spell is taken. */
    @Test
    void aSingleLegalOptionIsTaken() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        Card bears = inHand(game, a, "Grizzly Bears");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();

        assertNotNull(chosen, "a random draw never passes while anything is playable");
        assertEquals(1, chosen.size());
        assertEquals(bears, chosen.get(0).getHostCard());
        assertTrue(seat.lastDrawWasRandom());
    }

    /**
     * Scenario 15's first half, end to end: the drawn play is cast with the
     * AI paying its mana, and lands on the stack.
     */
    @Test
    void aRandomlyDrawnSpellIsCastOntoTheStack() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        inHand(game, a, "Grizzly Bears");
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
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        Card land = inHand(game, a, "Forest");
        inHand(game, a, "Grizzly Bears");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        List<SpellAbility> first = seat.chooseSpellAbilityToPlay();
        assertFalse(seat.lastDrawWasRandom(), "no draw while a land drop is open");
        assertNotNull(first, "the AI plays the land");
        assertTrue(first.get(0).isLandAbility(), first.toString());
        assertEquals(land, first.get(0).getHostCard());

        a.playLand(land, null);
        assertEquals(1, a.getLandsPlayedThisTurn());

        List<SpellAbility> second = seat.chooseSpellAbilityToPlay();
        assertTrue(seat.lastDrawWasRandom(), "the drop is used: the draw is made");
        assertNotNull(second);
        assertEquals("Grizzly Bears", second.get(0).getHostCard().getName());
    }

    /** Scenario 2's other branch: a draw that fails leaves the decision to the AI. */
    @Test
    void aFailedDrawLeavesThePlayToTheAi() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 0.0, 1L);
        Player a = game.getPlayers().get(0);
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        inHand(game, a, "Grizzly Bears");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        for (int i = 0; i < 20; i++) {
            seat.chooseSpellAbilityToPlay();
            assertFalse(seat.lastDrawWasRandom(), "P = 0 never draws");
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
                inPlay(game, a, "Mountain");
            }
            inHand(game, a, "Fiery Intervention");
            inHand(game, a, "Shock");
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
     * And when nothing remains the seat passes: the abandoned play returns
     * false, the engine asks again, and the answer is a pass rather than
     * another draw over the same impossible candidate.
     */
    @Test
    void whenEveryCandidateIsAbandonedTheSeatPasses() {
        Game game = gameWithRandomSeatA(PhaseType.MAIN2, 1.0, 1L);
        Player a = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            inPlay(game, a, "Mountain");
        }
        inHand(game, a, "Fiery Intervention");
        RandomSeatController seat = seatOf(game);
        seat.attach(game, null);

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
        assertNotNull(chosen, "the charm passes the candidate filter");
        assertFalse(seat.playChosenSpellAbility(chosen.get(0)));
        assertTrue(game.getStack().isEmpty());
        assertNull(seat.chooseSpellAbilityToPlay(), "nothing left to draw: pass");
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
                inPlay(game, a, "Mountain");
            }
            inHand(game, a, "Fiery Confluence");
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
        inPlay(game, a, "Forest");
        inPlay(game, a, "Forest");
        inHand(game, a, "Grizzly Bears");
        seatOf(game).attach(game, null);

        Game fork = new GameCopier(game).makeCopy();
        RandomSeatController forked = GamePlayer.randomSeatOf(fork);
        assertNotNull(forked, "the fork keeps the lobby-player subclass");
        assertFalse(forked.isLive());

        forked.chooseSpellAbilityToPlay();
        assertFalse(forked.lastDrawWasRandom(), "a fork's seat defers to the AI");
    }

    // ── attacks and blocks ──────────────────────────────────────────────

    /**
     * Scenario 2 and 16: with nothing restricting attacks, each creature
     * attacks in close to half the draws, and each declaration writes an
     * {@code attackers} record stamped a real decision by the seat itself.
     */
    @Test
    void eachCreatureAttacksInAboutHalfTheDrawsAndTheDeclarationIsRecorded() throws IOException {
        Game game = gameWithRandomSeatA(PhaseType.COMBAT_DECLARE_ATTACKERS, 1.0, 7L);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        List<Card> creatures = List.of(
                inPlay(game, a, "Grizzly Bears"), inPlay(game, a, "Runeclaw Bear"),
                inPlay(game, a, "Hill Giant"), inPlay(game, a, "Grizzly Bears"));
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        RandomSeatController seat = seatOf(game);
        int draws = 400;
        int[] attacks = new int[creatures.size()];
        try (PatchedCollectors collectors = new PatchedCollectors(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
            collectors.setRandomSeatPlayerId(SnapshotBuilder.playerId(a));
            seat.attach(game, collectors);
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
        List<String> records = readShard(path).stream()
                .filter(l -> l.contains("\"subkind\":\"attackers\"")).toList();
        assertFalse(records.isEmpty(), "the seat writes its own attackers record");
        for (String record : records) {
            assertTrue(record.contains("\"what_if\":false"), record);
            assertTrue(record.contains("\"random_seat\":true"), record);
            assertTrue(record.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(a) + "\""),
                    record);
        }
    }

    /**
     * Scenario 17: the random seat's own block declaration, whichever way it
     * draws, writes {@code blockers} records naming the seat as the deciding
     * player and stamped {@code random_seat}; the blocks it declares are legal.
     */
    @Test
    void aBlockDeclarationNamesTheBlockingSeat() throws IOException {
        Game game = gameWithRandomSeatA(PhaseType.COMBAT_DECLARE_BLOCKERS, 1.0, 3L);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        // B's turn: B attacks A with one creature, A has two it could block with.
        game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DECLARE_BLOCKERS, b);
        Card attacker = inPlay(game, b, "Hill Giant");
        List<Card> blockers = List.of(inPlay(game, a, "Grizzly Bears"), inPlay(game, a, "Runeclaw Bear"));
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        RandomSeatController seat = seatOf(game);
        boolean blockedOnce = false;
        boolean heldBackOnce = false;
        try (PatchedCollectors collectors = new PatchedCollectors(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
            collectors.setRandomSeatPlayerId(SnapshotBuilder.playerId(a));
            seat.attach(game, collectors);
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
        List<String> records = readShard(path).stream()
                .filter(l -> l.contains("\"subkind\":\"blockers\"")).toList();
        assertFalse(records.isEmpty());
        for (String record : records) {
            assertTrue(record.contains("\"what_if\":false"), record);
            assertTrue(record.contains("\"random_seat\":true"), record);
            assertTrue(record.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(a) + "\""),
                    record);
        }
    }

    // ── the match-level draw ────────────────────────────────────────────

    /** Scenario 1: one match in eight seats a random player, A or B alike. */
    @Test
    void closeToOneMatchInEightSeatsARandomPlayerOnEitherSide() {
        CollectionCaps caps = new CollectionCaps(
                1, 0.1, 2, 2, List.of(), List.of(1, 2, 3), 0.1, 0.125, 0.5);
        Random random = new Random(11L);
        int draws = 40_000;
        int a = 0;
        int b = 0;
        for (int i = 0; i < draws; i++) {
            switch (GamePlayer.drawRandomSeat(caps, random)) {
                case A -> a++;
                case B -> b++;
                default -> { }
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
            assertEquals(GamePlayer.RandomSeat.NONE,
                    GamePlayer.drawRandomSeat(CollectionCaps.defaults(), random));
        }
    }
}
