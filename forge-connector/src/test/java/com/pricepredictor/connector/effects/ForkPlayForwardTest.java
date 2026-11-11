package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.GamePlayer;
import com.pricepredictor.connector.RandomSeatController;
import com.pricepredictor.connector.RandomSeatLobbyPlayer;
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

    @TempDir
    Path tempDir;

    private static final Consumer<EffectRecord> DIRECTED = record -> record.fork(true).interventional(false);

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
            Forks.ForkRun run = new Forks.ForkRun(
                    live, () -> { }, () -> live.getPhaseHandler().getPlayerTurn());
            played = run.take(
                    (verified, snapshots) -> recording.playForward().play(verified.fork(), flags),
                    (verified, result) -> result);
            liveTurnAfterFork = live.getPhaseHandler().getTurn();
            livePhaseAfterFork = live.getPhaseHandler().getPhase();
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
        try (GameRecording recording = new GameRecording(game, writer, ShardRun.GAME_ID, caps, 7L, null)) {
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
                game, writer, ShardRun.GAME_ID, caps, 7L, null, throwing)) {
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
                game, writer, ShardRun.GAME_ID, caps, 7L, Refs.of(game.getPlayers().get(0)), seats)) {
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

    @Test
    void forkLinesAreHeldUntilTheForkIsKept() {
        List<String> held = new ArrayList<>();
        CollectorContext live = TestCollectors.context(ShardRun.writer(tempDir), ShardRun.GAME_ID, null);
        CollectorContext fork = live.forFork(DIRECTED, held::add);

        fork.emit(fork.newRecord(RecordKind.COMBAT).actor("P0").state("{}").payload("{}"));

        assertEquals(1, held.size());
        assertTrue(isFork(held.get(0)));
        assertEquals(Boolean.FALSE, parse(held.get(0)).get("interventional"));
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
        try (GameRecording recording = new GameRecording(game, writer, ShardRun.GAME_ID, caps, 7L, null)) {
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
}
