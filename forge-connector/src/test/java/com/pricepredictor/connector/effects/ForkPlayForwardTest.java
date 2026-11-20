package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.GamePlayer;
import com.pricepredictor.connector.RandomSeatController;
import com.pricepredictor.connector.RandomSeatLobbyPlayer;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import forge.ai.LobbyPlayerAi;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A fork taken at a phase event plays on to the end of the turn, with both
 * seats choosing, and is recorded by collectors of its own (D4).
 *
 * <p>The board: Sneak Attack has put a Hill Giant in on the active player's
 * third turn, so the end step sacrifices it, and the opponent has an empty
 * board, so the active player attacks. Each test forks the live game at a
 * phase event of that turn, as the directed-fork scheduler will, plays the
 * fork on, then plays the live turn out and reads the shard both wrote to.
 */
@ExtendWith(ForgeExtension.class)
class ForkPlayForwardTest {

    /**
     * A cleanup step in which a player got priority is taken again (514.3a),
     * and its turn-based actions are the turn's: the fork does not stop
     * before it, as it stops before a cleanup that ends the turn.
     */
    @Test
    void aCleanupToBeTakenAgainDoesNotEndTheTurn() throws ReflectiveOperationException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.CLEANUP);
        assertTrue(ForkPlayForward.turnWouldEnd(game), "an ordinary cleanup ends the turn");
        java.lang.reflect.Field repeat =
                forge.game.phase.PhaseHandler.class.getDeclaredField("bRepeatCleanup");
        repeat.setAccessible(true);
        repeat.setBoolean(game.getPhaseHandler(), true);

        assertFalse(ForkPlayForward.turnWouldEnd(game));
    }


    @TempDir
    Path tempDir;

    private static final Consumer<EffectRecord> DIRECTED = record -> record.fork(ForkKind.DIRECTED).interventional(false);

    static Game board(Game game) {
        game.setAge(GameStage.Play);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, active, 3);
        for (int i = 0; i < 5; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, active, "Forest"), null, null);
            game.getAction().moveToLibrary(TestBoard.card(game, opponent, "Forest"), null, null);
        }
        TestBoard.inPlay(game, active, "Sneak Attack");
        for (int i = 0; i < 3; i++) {
            TestBoard.inPlay(game, active, "Mountain");
            TestBoard.inPlay(game, opponent, "Forest");
        }
        TestBoard.inPlay(game, active, "Grizzly Bears");
        TestBoard.inHand(game, active, "Hill Giant");
        return game;
    }

    static Game board() {
        return board(TestBoard.twoPlayerGame(PhaseType.MAIN1));
    }

    static Card sneakAttack(Game game) {
        Player active = game.getPlayers().get(0);
        Card sneak = onBattlefield(game, "Sneak Attack");
        SpellAbility put = TestBoard.ability(sneak, ApiType.ChangeZone);
        put.setActivatingPlayer(active);
        TestBoard.castAndResolve(game, put);
        return onBattlefield(game, "Hill Giant");
    }

    private static Card onBattlefield(Game game, String name) {
        return game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(name)).findFirst().orElseThrow();
    }

    /** Forks the live game once, at the first phase event of {@code phase}, and plays the fork on. */
    private static final class Forker {
        private final Game live;
        private final GameRecording recording;
        private final PhaseType phase;
        private final Consumer<EffectRecord> flags;
        ForkPlayForward.Played played;
        boolean taken;
        int liveTurnAfterFork = -1;
        PhaseType livePhaseAfterFork;
        /** The live board as BoardStates reads it, before the fork and after it played on. */
        Object liveBoardBefore;
        Object liveBoardAfter;

        Forker(Game live, GameRecording recording, PhaseType phase, Consumer<EffectRecord> flags) {
            this.live = live;
            this.recording = recording;
            this.phase = phase;
            this.flags = flags;
        }

        @Subscribe
        public void onPhase(GameEventTurnPhase event) {
            if (taken || event.phase() != phase) {
                return;
            }
            taken = true;
            liveBoardBefore = BoardStates.of(live);
            Forks.ForkRun run = new Forks.ForkRun(
                    "directed", live, () -> { }, () -> live.getPhaseHandler().getPlayerTurn());
            played = run.take(
                    (verified, snapshots) -> recording.playForward().play(verified.fork(), flags),
                    (verified, result) -> result);
            liveTurnAfterFork = live.getPhaseHandler().getTurn();
            livePhaseAfterFork = live.getPhaseHandler().getPhase();
            liveBoardAfter = BoardStates.of(live);
        }
    }

    static void playTheTurnOut(Game game) {
        int turn = game.getPhaseHandler().getTurn();
        for (int steps = 0; !game.isGameOver() && game.getPhaseHandler().getTurn() == turn; steps++) {
            if (steps > ForkPlayForward.MAX_STEPS) {
                throw new AssertionError("the live turn did not end");
            }
            game.getPhaseHandler().mainLoopStep();
        }
    }

    /** Record the live game, fork it at {@code phase}, play both out; the shard's lines. */
    private List<String> forkAt(Game game, PhaseType phase, Consumer<EffectRecord> flags, Forker[] out)
            throws Throwable {
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).everyDecision().build();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            Forker forker = new Forker(game, recording, phase, flags);
            out[0] = forker;
            game.subscribeToEvents(forker);
            playTheTurnOut(game);
        } finally {
            writer.close();
        }
        return ShardRun.readShard(writer.path());
    }

    private static boolean isFork(String record) {
        return Boolean.TRUE.equals(parse(record).get("fork"));
    }

    private static List<String> forkRecords(List<String> all) {
        return all.stream().filter(ForkPlayForwardTest::isFork).toList();
    }

    private static List<String> liveRecords(List<String> all) {
        return all.stream().filter(record -> !isFork(record)).toList();
    }

    static String deathOf(Card card) {
        return "{\"type\":\"zone_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\"";
    }

    private static long naming(List<String> records, String needle) {
        return records.stream().filter(record -> record.contains(needle)).count();
    }

    // ── playing on ──────────────────────────────────────────────────────

    /**
     * The reproducer: forked as the end step begins, the fork begins that
     * step itself, so Sneak Attack's sacrifice resolves in it, and its
     * records say so beside the live game's own.
     */
    @Test
    void aForkAtTheEndStepResolvesTheDelayedSacrifice() throws Throwable {
        Game game = board();
        Card giant = sneakAttack(game);
        Forker[] forker = new Forker[1];

        List<String> all = forkAt(game, PhaseType.END_OF_TURN, DIRECTED, forker);

        assertNotNull(forker[0].played, "the fork was played out");
        assertFalse(forker[0].played.gameOver());
        assertEquals(1, naming(forkRecords(all), deathOf(giant)),
                "the fork's own records hold its sacrifice");
        assertEquals(1, naming(liveRecords(all), deathOf(giant)),
                "and the live game's hold its own");
        assertEquals(PhaseType.END_OF_TURN, forker[0].livePhaseAfterFork,
                "the live game had not moved while the fork played");
        assertEquals(3, forker[0].liveTurnAfterFork);
        // Nor had its board: every card, player and member as before.
        assertEquals(forker[0].liveBoardBefore, forker[0].liveBoardAfter,
                "the live board moved while the fork played");
    }

    /**
     * Forked as attackers are about to be declared, the fork declares its
     * own, and its combat and its decisions are recorded by its collectors.
     */
    @Test
    void aForkAtTheDeclareAttackersStepAttacksAndRecordsItsCombat() throws Throwable {
        Game game = board();
        sneakAttack(game);
        Forker[] forker = new Forker[1];

        List<String> all = forkAt(game, PhaseType.COMBAT_DECLARE_ATTACKERS, DIRECTED, forker);
        List<String> forked = forkRecords(all);

        assertNotNull(forker[0].played);
        assertTrue(forked.stream().anyMatch(record -> "combat".equals(parse(record).get("kind"))),
                "the fork's damage step wrote a combat record");
        assertTrue(forked.stream().anyMatch(record -> "playability".equals(parse(record).get("kind"))),
                "the hooks routed the fork's decisions to its own collectors");
        for (String record : forked) {
            assertEquals(Boolean.FALSE, parse(record).get("interventional"), record);
            assertEquals(ShardRun.GAME_ID, parse(record).get("game_id"), "a fork joins its live game by id");
        }
        assertEquals(liveRecords(all).stream().filter(r -> "combat".equals(parse(r).get("kind"))).count(),
                forked.stream().filter(r -> "combat".equals(parse(r).get("kind"))).count(),
                "both attacked once, each writing its own combat");
    }

    /**
     * A fork whose engine throws part way leaves no record behind, though
     * its collectors had written some: its seat, asked for a play at its
     * first priority, throws, after the declaration and the priority stop
     * were recorded.
     */
    @Test
    void aForkThatThrowsPartWayWritesNothing() throws Throwable {
        Game game = board();
        sneakAttack(game);
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).everyDecision().build();
        List<String> heldBeforeTheThrow = new ArrayList<>();
        ForkSeats throwing = (fork, candidates, fruitless, random) -> {
            Player seat = fork.getPlayers().get(0);
            seat.dangerouslySetController(new PlayerControllerAi(fork, seat, seat.getLobbyPlayer()) {
                @Override
                public List<SpellAbility> chooseSpellAbilityToPlay() {
                    throw new IllegalStateException("a card this fork cannot describe");
                }
            });
        };
        Consumer<EffectRecord> counting = DIRECTED.andThen(record -> heldBeforeTheThrow.add(record.toJson()));
        RecordShardWriter writer = ShardRun.writer(tempDir);
        Forker forker;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L)
                        .withForkSeats(throwing))) {
            recording.open();
            forker = new Forker(game, recording, PhaseType.COMBAT_DECLARE_ATTACKERS, counting);
            game.subscribeToEvents(forker);
            playTheTurnOut(game);
        } finally {
            writer.close();
        }
        List<String> all = ShardRun.readShard(writer.path());

        assertTrue(forker.taken);
        assertNull(forker.played, "the guard discarded the fork");
        assertFalse(heldBeforeTheThrow.isEmpty(), "the fork had recorded before it threw");
        assertTrue(forkRecords(all).isEmpty(), "and none of it was written");
        assertFalse(liveRecords(all).isEmpty(), "the live game recorded on");
    }

    /**
     * A fork whose attack wins the game writes what the live game would:
     * its lethal combat record, and the game's outcome (C1). Forge publishes
     * the game's end only from a match, which a fork has none of.
     */
    @Test
    void aForkThatEndsTheGameWritesItsLethalCombatAndTheOutcome() throws Throwable {
        Game game = board();
        game.getPlayers().get(1).setLife(1, null);
        Forker[] forker = new Forker[1];

        List<String> forked = forkRecords(
                forkAt(game, PhaseType.COMBAT_DECLARE_ATTACKERS, DIRECTED, forker));

        assertNotNull(forker[0].played, "the fork was played out");
        assertTrue(forker[0].played.gameOver(), "the fork's attack won the game");
        assertTrue(forked.stream().anyMatch(record -> "combat".equals(parse(record).get("kind"))),
                "the lethal damage step wrote its combat record: " + forked);
        assertEquals(1, RecordLines.withEventOfType(forked, "player_won").size(),
                "the outcome reached the fork's records: " + forked);
    }

    /**
     * A fork played on from the second main phase stops as its turn ends
     * (F10): nothing of the next turn is recorded, neither its turn-based
     * actions nor a record read at its first step's boundary. The Bears'
     * pump wears off in the cleanup step, so a Glorious Anthem's board
     * differs at the next turn's untap boundary and would be recorded there.
     */
    @Test
    void aForkPlayedFromTheSecondMainPhaseStopsAtTheEndOfItsTurn() throws Throwable {
        Game game = board();
        sneakAttack(game);
        Player active = game.getPlayers().get(0);
        TestBoard.inPlay(game, active, "Glorious Anthem");
        Card growth = TestBoard.inHand(game, active, "Giant Growth");
        SpellAbility pump = TestBoard.spellOf(game.getAction().moveToStack(growth, null), active);
        pump.resetTargets();
        pump.getTargets().add(onBattlefield(game, "Grizzly Bears"));
        TestBoard.castAndResolve(game, pump);
        Forker[] forker = new Forker[1];

        List<String> forked = forkRecords(forkAt(game, PhaseType.MAIN2, DIRECTED, forker));

        assertNotNull(forker[0].played, "the fork was played out");
        assertFalse(forked.isEmpty(), "the fork recorded its turn");
        for (String record : forked) {
            Object subkind = parse(record).get("subkind");
            assertFalse(subkind != null && List.of("untap", "phasing", "day_night").contains(subkind),
                    record);
            Object turn = RecordLines.asMap(RecordLines.asMap(parse(record).get("state")).get("global"))
                    .get("turn");
            assertEquals(3L, turn, "a record of the next turn: " + record);
        }
    }

    // ── the seats ───────────────────────────────────────────────────────

    /**
     * The fork's random seat is attached to the fork with draws of its own,
     * and the match's source gives it none: the next game's seat draws what
     * it would have drawn without the fork.
     */
    @Test
    void theForksRandomSeatDrawsFromTheForkSeedNotTheMatchSource() throws Throwable {
        Random seeds = new Random(42L);
        Random expected = new Random(42L);
        Game game = board(TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 0.0, seeds), new LobbyPlayerAi("b", null)));
        Card giant = sneakAttack(game);
        RandomSeatController liveSeat = (RandomSeatController) game.getPlayers().get(0).getController();
        AtomicReference<Game> attachedTo = new AtomicReference<>();
        ForkSeats seats = (fork, candidates, fruitless, random) -> {
            attachedTo.set(fork);
            GamePlayer.attachForkSeats(fork, candidates, fruitless, random);
        };
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).build();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        List<String> all;
        Forker forker;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L)
                        .withRandomSeat(Refs.of(game.getPlayers().get(0)))
                        .withForkSeats(seats))) {
            recording.open();
            liveSeat.attach(game, recording.priorityCandidates(), recording.fruitlessActivations());
            expected.nextLong();
            forker = new Forker(game, recording, PhaseType.END_OF_TURN, DIRECTED);
            game.subscribeToEvents(forker);
            playTheTurnOut(game);
        } finally {
            writer.close();
        }
        all = ShardRun.readShard(writer.path());

        assertNotNull(forker.played);
        assertNotNull(attachedTo.get(), "the fork's seats were attached");
        assertTrue(attachedTo.get() != game, "to the fork, not the live game");
        assertTrue(attachedTo.get().getPlayers().get(0).getController() instanceof RandomSeatController,
                "the fork's copy of the seat is a random seat");
        assertEquals(expected.nextLong(), seeds.nextLong(), "the fork drew nothing from the match's source");
        assertEquals(1, naming(forkRecords(all), deathOf(giant)));
    }

    // ── the fork's floating effects (B4) ────────────────────────────────

    /** The timestamped floating effects a record's global block lists. */
    private static List<String> timestampedIn(String record) {
        Object global = RecordLines.asMap(parse(record).get("state")).get("global");
        List<String> ids = new ArrayList<>();
        for (Object each : (List<?>) RecordLines.asMap(global).get("floating_effects")) {
            String id = (String) RecordLines.asMap(each).get("id");
            if (id.startsWith("F")) {
                ids.add(id);
            }
        }
        return ids;
    }

    /** Whether a record's board shows {@code card} with a +3/+3 temporary boost. */
    private static boolean pumped(String record, Card card) {
        for (Object each : (List<?>) RecordLines.asMap(parse(record).get("state")).get("entities")) {
            Map<String, Object> entity = RecordLines.asMap(each);
            if (("E" + card.getId()).equals(entity.get("id")) && entity.get("pt_boosts") != null) {
                return List.of(3L, 3L).equals(
                        RecordLines.asMap(entity.get("pt_boosts")).get("temporary"));
            }
        }
        return false;
    }

    /**
     * A Giant Growth resolved in the first main phase is a floating effect
     * {@code F<ts>} of the live game's records; a fork taken at the declare
     * attackers step lists it too, on every record whose board shows the
     * pump: the fork's clause notes start from the live game's (B4).
     */
    @Test
    void aForksRecordsListAPumpResolvedBeforeTheFork() throws Throwable {
        Game game = board();
        sneakAttack(game);
        Player active = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears");
        SpellAbility growth = TestBoard.spellOf(TestBoard.inHand(game, active, "Giant Growth"), active);
        growth.resetTargets();
        growth.getTargets().add(bears);
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).everyDecision().build();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        Forker forker;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            game.getStack().addAndUnfreeze(growth);
            game.getStack().resolveStack();
            forker = new Forker(game, recording, PhaseType.COMBAT_DECLARE_ATTACKERS, DIRECTED);
            game.subscribeToEvents(forker);
            playTheTurnOut(game);
        } finally {
            writer.close();
        }
        List<String> all = ShardRun.readShard(writer.path());

        assertNotNull(forker.played, "the fork was played out");
        List<String> live = liveRecords(all).stream().filter(r -> pumped(r, bears)).toList();
        assertFalse(live.isEmpty(), "a live record shows the pump");
        String pump = timestampedIn(live.get(0)).get(0);
        List<String> forked = forkRecords(all).stream().filter(r -> pumped(r, bears)).toList();
        assertFalse(forked.isEmpty(), "a fork record shows the pump: " + forkRecords(all));
        for (String record : forked) {
            assertTrue(timestampedIn(record).contains(pump), record);
        }
    }

    /**
     * FR-010d: Karn Liberated at 14 loyalty as a fork's first main phase
     * begins, and a seat that activates its restart at its first priority.
     * The ability reaches the fork's stack, and the fork is refused as
     * {@code game restart} before it resolves: the JVM-wide trigger id
     * counter is not lowered, nothing counts the live game as moved, and the
     * live Karn keeps its loyalty. The live game is stepped only up to the
     * boundary, so its own AI never gets to activate the restart.
     */
    @Test
    void aForkWhoseStackHoldsARestartIsRefusedBeforeItResolves() throws Throwable {
        Game game = board();
        Player active = game.getPlayers().get(0);
        game.getPhaseHandler().devModeSet(PhaseType.UPKEEP, active, 3);
        Card karn = TestBoard.inPlay(game, active, "Karn Liberated");
        karn.setCounters(forge.game.card.CounterEnumType.LOYALTY, 14);
        AtomicReference<SpellAbility> activated = new AtomicReference<>();
        ForkSeats restarting = (fork, candidates, fruitless, random) -> {
            Player seat = fork.getPlayers().get(0);
            seat.dangerouslySetController(new PlayerControllerAi(fork, seat, seat.getLobbyPlayer()) {
                @Override
                public List<SpellAbility> chooseSpellAbilityToPlay() {
                    if (activated.get() != null || !fork.getPhaseHandler().is(PhaseType.MAIN1)) {
                        return null;
                    }
                    Card forked = fork.getCardsIn(ZoneType.Battlefield).stream()
                            .filter(card -> card.getName().equals("Karn Liberated")).findFirst().orElseThrow();
                    SpellAbility restart = TestBoard.ability(forked, ApiType.RestartGame);
                    restart.setActivatingPlayer(seat);
                    activated.set(restart);
                    return List.of(restart);
                }
            });
        };
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).build();
        String refusal = "directed " + ForkRefusal.GAME_RESTART;
        long refusedBefore = ForkRefusals.counts().getOrDefault(refusal, 0L);
        long movedBefore = HookFailures.counts().getOrDefault("fork directed moved the live game", 0L);
        int triggerIdsBefore = Forks.LiveMarks.currentTriggerIds();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        Forker forker;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L)
                        .withForkSeats(restarting))) {
            recording.open();
            forker = new Forker(game, recording, PhaseType.MAIN1, DIRECTED);
            game.subscribeToEvents(forker);
            for (int steps = 0; !forker.taken; steps++) {
                assertTrue(steps < ForkPlayForward.MAX_STEPS, "the live game reached its first main phase");
                game.getPhaseHandler().mainLoopStep();
            }
        } finally {
            writer.close();
        }

        assertNotNull(activated.get(), "the fork's seat activated the restart");
        assertNull(forker.played, "the fork was refused");
        assertEquals(refusedBefore + 1, ForkRefusals.counts().getOrDefault(refusal, 0L));
        assertTrue(Forks.LiveMarks.currentTriggerIds() >= triggerIdsBefore, "the trigger id counter was not reset");
        assertEquals(movedBefore, HookFailures.counts().getOrDefault("fork directed moved the live game", 0L));
        assertEquals(14, karn.getCounters(forge.game.card.CounterEnumType.LOYALTY));
        assertTrue(game.getAge() != GameStage.RestartedByKarn);
        assertTrue(forkRecords(ShardRun.readShard(writer.path())).isEmpty(), "the refused fork wrote nothing");
    }

    // ── the premise ─────────────────────────────────────────────────────

    /**
     * A directed fork's edit that leaves a creature at 0 toughness: it dies
     * in the state-based check the fork runs before it plays on (704.5f, a
     * creature with toughness 0 is put into its owner's graveyard), and the
     * death is in the fork's premise record, its first, a {@code fork_edit}
     * turn action. The rule is this test's own: no registered rule's edit has
     * a state-based consequence, since the twin every
     * {@code RegisteredRules*Test} forks through ({@code RuleBoard.twin})
     * requires the check after each edit to change nothing.
     */
    @Test
    void anEditsStateBasedDeathIsInTheForksPremiseRecord() throws Throwable {
        Game game = board();
        Card bears = onBattlefield(game, "Grizzly Bears");

        List<String> forked = zeroToughnessFork(game, bears);

        Map<String, Object> premise = parse(forked.get(0));
        assertEquals("turn_action", premise.get("kind"), forked.get(0));
        assertEquals("fork_edit", premise.get("subkind"), forked.get(0));
        assertTrue(forked.get(0).contains(deathOf(bears)), forked.get(0));
        assertEquals(1, naming(forked, deathOf(bears)), "the death is the premise's alone");
        for (String record : forked) {
            assertEquals("directed", parse(record).get("fork_kind"), record);
        }
    }

    /** A directed fork at the live main phase setting {@code creature}'s toughness to 0; its lines. */
    private List<String> zeroToughnessFork(Game game, Card creature) throws Exception {
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).build();
        Application zero = new Application(ForkOperation.PT,
                binding -> new PtOperation(binding.card("creature"), null, 0));
        DirectedRule rule = DirectedRule.of("704.5f", "704.5f-1", live -> Optional.empty(), new DirectedRule.Variant("zero", zero));
        RecordShardWriter writer = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            assertTrue(new DirectedFork(game, recording.playForward(), () -> game.getPhaseHandler().getPlayerTurn())
                    .take(rule, "zero", zero, Binding.empty().with("creature", creature)), "the fork was kept");
        } finally {
            writer.close();
        }
        return ShardRun.readShard(writer.path());
    }
}
