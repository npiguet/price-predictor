package com.pricepredictor.connector.effects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.util.MyRandom;
import java.util.Map;
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

        Object written = run("run-test-throws", live, spent).take((verified, snapshots) -> {
            throw new IllegalStateException("a card this fork cannot describe");
        }, (verified, produced) -> "written");

        assertNull(written);
        assertEquals(1, spent.get(), "a discarded fork still counts against its budget");
        assertEquals(before + 1, count("fork run-test-throws body"));
    }

    /** A body that overflows the stack is contained as one that throws, not the worker's end. */
    @Test
    void aForkWhoseBodyOverflowsTheStackIsCountedAndDiscarded() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        long before = count("fork run-test-overflow body");

        Object written = run("run-test-overflow", live, new AtomicInteger()).take((verified, snapshots) -> {
            throw new StackOverflowError();
        }, (verified, produced) -> "written");

        assertNull(written);
        assertEquals(before + 1, count("fork run-test-overflow body"));
    }

    /** A copy that throws is counted under its kind's copy, and nothing runs. */
    @Test
    void aForkWhoseCopyThrowsIsCounted() {
        long before = count("fork run-test-copy copy");
        AtomicInteger ran = new AtomicInteger();

        Object written = run("run-test-copy", null, new AtomicInteger()).take((verified, snapshots) -> {
            ran.incrementAndGet();
            return "ran";
        }, (verified, produced) -> produced);

        assertNull(written);
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
                    (verified, snapshots) -> MyRandom.getRandom().nextInt(1000),
                    (verified, produced) -> produced);

            assertTrue(drawn instanceof Integer, "the body did not run: " + drawn);
            assertSame(stream, MyRandom.getRandom(), "the live stream was not put back");
            assertEquals(0, stream.draws, "the fork drew from the live game's stream");
        } finally {
            MyRandom.setRandom(was);
        }
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
                (fork, snapshots) -> fork, (fork, produced) -> produced);

        assertEquals(Map.of("remembered", 1), verified.copyDropped());
        String stamped = verified.stamp(new EffectRecord(
                "r", "run", "t", "g", RecordKind.COMBAT, AttributionMode.PATCHED)).toJson();
        assertTrue(stamped.contains("\"copy_dropped\":{\"remembered\":1}"), stamped);
    }

    /** A copy that left nothing out stamps nothing: the member is absent. */
    @Test
    void aLosslessCopyStampsNothing() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(live, live.getPlayers().get(0), "Hill Giant");

        Forks.VerifiedFork verified = run("directed", live, new AtomicInteger()).take(
                (fork, snapshots) -> fork, (fork, produced) -> produced);

        assertTrue(verified.copyDropped().isEmpty(), verified.copyDropped().toString());
        assertFalse(verified.stamp(new EffectRecord(
                "r", "run", "t", "g", RecordKind.COMBAT, AttributionMode.PATCHED))
                .toJson().contains("copy_dropped"));
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
