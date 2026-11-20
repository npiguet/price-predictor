package com.pricepredictor.connector.effects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.StdStreams;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * One fork's life ({@link Forks.ForkRun}), on a started game: what it
 * counts when it fails, what it draws, and what it never touches.
 */
@ExtendWith(ForgeExtension.class)
class ForkRunTest {

    private static long count(String key) {
        return HookFailures.counts().getOrDefault(key, 0L);
    }

    private static Forks.ForkRun run(String kind, Game live, AtomicInteger spent) {
        return new Forks.ForkRun(kind, live, spent::incrementAndGet,
                () -> live == null ? null : live.getPhaseHandler().getPlayerTurn());
    }

    /**
     * A body that throws is discarded, after the budget was spent, and is
     * counted under its kind in the worker's failure report: a systematically
     * failing fork leaves a trace.
     */
    @Test
    void aForkWhoseBodyThrowsIsCountedAndDiscarded() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        AtomicInteger spent = new AtomicInteger();
        long before = count("fork run-test-throws body");

        Optional<?> written = run("run-test-throws", live, spent).take((verified, snapshots) -> {
            throw new IllegalStateException("a card this fork cannot describe");
        }, (verified, produced) -> "written");

        assertTrue(written.isEmpty());
        assertEquals(1, spent.get(), "a discarded fork still counts against its budget");
        assertEquals(before + 1, count("fork run-test-throws body"));
    }

    /** A body that overflows the stack is contained as one that throws, not the worker's end. */
    @Test
    void aForkWhoseBodyOverflowsTheStackIsCountedAndDiscarded() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        long before = count("fork run-test-overflow body");

        Optional<?> written = run("run-test-overflow", live, new AtomicInteger()).take((verified, snapshots) -> {
            throw new StackOverflowError();
        }, (verified, produced) -> "written");

        assertTrue(written.isEmpty());
        assertEquals(before + 1, count("fork run-test-overflow body"));
    }

    /** A copy that throws is counted under its kind's copy, and nothing runs. */
    @Test
    void aForkWhoseCopyThrowsIsCounted() throws Throwable {
        long before = count("fork run-test-copy copy");
        AtomicInteger ran = new AtomicInteger();
        Optional<?>[] written = new Optional<?>[1];

        // Its first failure prints a line with the stack: captured, not shown.
        String printed = StdStreams.captureStderr(() -> written[0] = run("run-test-copy", null, new AtomicInteger())
                .take((verified, snapshots) -> {
                    ran.incrementAndGet();
                    return Optional.of("ran");
                }, (verified, produced) -> produced));

        assertTrue(written[0].isEmpty());
        assertTrue(printed.isEmpty() || printed.contains("fork run-test-copy copy threw"), printed);
        assertEquals(0, ran.get());
        assertEquals(before + 1, count("fork run-test-copy copy"));
    }

    /**
     * Whatever a fork draws, the copy, its check and its body alike, comes
     * from a stream of its own: the live game's stream is put back untouched.
     */
    @Test
    void aForkDrawsNothingFromTheLiveGamesRandomStream() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(live, live.getPlayers().get(0), "Grizzly Bears");
        CountingRandom stream = new CountingRandom();
        Random was = MyRandom.getRandom();
        MyRandom.setRandom(stream);
        try {
            Object drawn = run("directed", live, new AtomicInteger()).take(
                    (verified, snapshots) -> Optional.of(MyRandom.getRandom().nextInt(1000)),
                    (verified, produced) -> produced).orElse(null);

            assertTrue(drawn instanceof Integer, "the body did not run: " + drawn);
            assertSame(stream, MyRandom.getRandom(), "the live stream was not put back");
            assertEquals(0, stream.draws, "the fork drew from the live game's stream");
        } finally {
            MyRandom.setRandom(was);
        }
    }

    private static long firstForkDraw(long gameSeed) {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(live, live.getPlayers().get(0), "Grizzly Bears");
        SeedStreams.register(live, gameSeed);
        Object drawn = run("directed", live, new AtomicInteger()).take(
                (verified, snapshots) -> Optional.of(MyRandom.getRandom().nextLong()), (verified, produced) -> produced)
                .orElse(null);
        assertTrue(drawn instanceof Long, "the body did not run: " + drawn);
        return (Long) drawn;
    }

    /**
     * A fork's stream is its game's next, seeded from the game's seed: the
     * first fork of two games draws differently, and of a game seeded again
     * the same.
     */
    @Test
    void theFirstForkOfTwoGamesDrawsDifferentSequences() {
        assertEquals(firstForkDraw(41L), firstForkDraw(41L));
        assertTrue(firstForkDraw(41L) != firstForkDraw(42L), "two games' first forks drew alike");
    }

    /**
     * What the copier left out is read off the copy and handed to every
     * record of the fork as {@code copy_dropped}: a card remembering a token
     * that has ceased to exist leaves one remembered object out.
     */
    @Test
    void aLossyCopyIsStampedOnTheForksRecords() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player player = live.getPlayers().get(0);
        Card holder = TestBoard.inPlay(live, player, "Hill Giant");
        Card token = TestBoard.inPlay(live, player, "Grizzly Bears");
        holder.addRemembered(token);
        token.getZone().remove(token);

        Forks.VerifiedFork verified = run("directed", live, new AtomicInteger()).take(
                (fork, snapshots) -> Optional.of(fork), (fork, produced) -> produced).orElseThrow();

        assertEquals(Map.of("remembered", 1), verified.copyDropped());
        String stamped = verified.stamp(new EffectRecord(
                new RecordHeader("r", "run", "t", "g"), RecordKind.COMBAT,
                AttributionMode.PATCHED)).toJson();
        assertTrue(stamped.contains("\"copy_dropped\":{\"remembered\":1}"), stamped);
    }

    /** A copy that left nothing out stamps nothing: the member is absent. */
    @Test
    void aLosslessCopyStampsNothing() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(live, live.getPlayers().get(0), "Hill Giant");

        Forks.VerifiedFork verified = run("directed", live, new AtomicInteger()).take(
                (fork, snapshots) -> Optional.of(fork), (fork, produced) -> produced).orElseThrow();

        assertTrue(verified.copyDropped().isEmpty(), verified.copyDropped().toString());
        assertFalse(verified.stamp(new EffectRecord(
                new RecordHeader("r", "run", "t", "g"), RecordKind.COMBAT,
                AttributionMode.PATCHED))
                .toJson().contains("copy_dropped"));
    }

    /**
     * A fork that changes the live game is counted under its kind: here a
     * body that takes a timestamp from the live game rather than its copy,
     * the way Forge's evaluator once did.
     */
    @Test
    void aForkThatMovesTheLiveGameIsCounted() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        long before = count("fork run-test-leak moved the live game");

        Object written = run("run-test-leak", live, new AtomicInteger()).take(
                (verified, snapshots) -> Optional.of(live.getNextTimestamp()), (verified, produced) -> produced)
                .orElse(null);

        assertTrue(written instanceof Long, "the fork ran: " + written);
        assertEquals(before + 1, count("fork run-test-leak moved the live game"));
    }

    /**
     * A fork whose game is restarted (Karn Liberated's ultimate resolving in
     * it) resets Forge's JVM-wide trigger id counter, which numbers the live
     * game's later triggers: it is counted as having moved the live game.
     * The counter is put back above every id handed out, so later tests make
     * no trigger an earlier one's id.
     */
    @Test
    void aForkThatRestartsItsGameMovesTheLiveGame() throws ReflectiveOperationException {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPhaseHandler().getPlayerTurn();
        TestBoard.inPlay(live, you, "Karn Liberated").setCounters(forge.game.card.CounterEnumType.LOYALTY, 14);
        java.lang.reflect.Field maxId = forge.game.trigger.Trigger.class.getDeclaredField("maxId");
        maxId.setAccessible(true);
        int ids = Math.max(maxId.getInt(null), 60000);
        maxId.setInt(null, ids);
        long before = count("fork run-test-restart moved the live game");
        try {
            Object written = run("run-test-restart", live, new AtomicInteger()).take((verified, snapshots) -> {
                Card karn = verified.fork().getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                        .filter(card -> card.getName().equals("Karn Liberated")).findFirst().orElseThrow();
                SpellAbility restart = karn.getSpellAbilities().stream()
                        .filter(sa -> sa.getApi() == forge.game.ability.ApiType.RestartGame).findFirst().orElseThrow();
                restart.setActivatingPlayer(karn.getController());
                forge.game.ability.AbilityUtils.resolve(restart);
                return Optional.of(verified.fork().getAge());
            }, (verified, produced) -> produced).orElse(null);

            assertEquals(forge.game.GameStage.RestartedByKarn, written, "the fork's game restarted");
            assertTrue(Forks.LiveMarks.currentTriggerIds() < ids, "the counter went down");
            assertEquals(before + 1, count("fork run-test-restart moved the live game"));
        } finally {
            // Just above what was handed out: far below Characterization.PinnedForgeIds' base,
            // which refuses to pin once the counter has reached it.
            maxId.setInt(null, Math.max(maxId.getInt(null), ids) + 1);
        }
    }

    /**
     * The score check plays the turn's combat out in copies of both games,
     * and leaves the live game's timestamp and controllers as they were:
     * nothing is counted as having moved it.
     */
    @Test
    void aForksScoreCheckLeavesTheLiveGameAsItWas() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPhaseHandler().getPlayerTurn();
        for (String name : List.of("Grizzly Bears", "Hill Giant")) {
            TestBoard.inPlay(live, you, name).setSickness(false);
        }
        TestBoard.inPlay(live, you.getWeakestOpponent(), "Gray Ogre");
        long stamp = live.getTimestamp();
        long before = count("fork run-test-check moved the live game");

        Object written = run("run-test-check", live, new AtomicInteger()).take(
                (verified, snapshots) -> Optional.of("checked"), (verified, produced) -> produced).orElse(null);

        assertEquals("checked", written);
        assertEquals(stamp, live.getTimestamp());
        assertEquals(before, count("fork run-test-check moved the live game"));
    }

    /**
     * A body that declines for an expected reason is a refusal, counted by
     * reason apart from the failures, and writes nothing.
     */
    @Test
    void aRefusalIsCountedApartFromTheFailures() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        long refused = ForkRefusals.counts().getOrDefault("run-test-refuse no choices", 0L);

        Optional<?> written = run("run-test-refuse", live, new AtomicInteger()).take((verified, snapshots) -> {
            throw new ForkRefusal(ForkRefusal.NO_CHOICES);
        }, (verified, produced) -> "written");

        assertTrue(written.isEmpty());
        assertEquals(refused + 1, ForkRefusals.counts().get("run-test-refuse no choices"));
        assertEquals(0, count("fork run-test-refuse body"));
    }

    /** A body that produces nothing writes nothing, and is neither a failure nor a refusal. */
    @Test
    void aBodyThatProducesNothingIsDiscardedUnwritten() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        AtomicInteger spent = new AtomicInteger();
        boolean[] written = {false};

        Optional<?> result = run("run-test-nothing", live, spent).take(
                (verified, snapshots) -> Optional.empty(), (verified, produced) -> {
                    written[0] = true;
                    return "written";
                });

        assertTrue(result.isEmpty());
        assertFalse(written[0], "the writer never ran");
        assertEquals(1, spent.get(), "a discarded fork still counts against its budget");
        assertEquals(0, count("fork run-test-nothing body"));
    }

    /** A writer that throws is counted under the fork's kind, not the boundary step's. */
    @Test
    void aWriterThatThrowsIsCountedUnderTheFork() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        long before = count("fork run-test-write write");

        Optional<?> written = run("run-test-write", live, new AtomicInteger()).take(
                (verified, snapshots) -> Optional.of("ran"), (verified, produced) -> {
                    throw new IllegalStateException("a record that cannot be written");
                });

        assertTrue(written.isEmpty());
        assertEquals(before + 1, count("fork run-test-write write"));
    }

    /** A random stream that counts what is drawn from it. */
    private static final class CountingRandom extends Random {
        int draws;

        CountingRandom() {
            super(1L);
        }

        @Override
        protected int next(int bits) {
            draws++;
            return super.next(bits);
        }
    }
}
