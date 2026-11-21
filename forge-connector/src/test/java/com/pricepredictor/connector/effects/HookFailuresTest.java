package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * What a failing hook call or bus subscriber loses is counted (C1 #2, V2
 * #20): a subscriber's exception, which Guava's {@code EventBus} logs and
 * swallows, and a hook call short of the arguments its listener reads, which
 * is answered with nothing.
 */
@ExtendWith(ForgeExtension.class)
class HookFailuresTest {

    @TempDir
    Path tempDir;

    /** A subscriber on a game's bus that throws on every phase event. */
    public static final class ThrowingSubscriber {
        @Subscribe
        public void onPhase(GameEventTurnPhase event) {
            throw new IllegalStateException("a phase this subscriber cannot read");
        }
    }

    /**
     * Each exception a subscriber throws is counted under its class and
     * method, once a game is recorded: the bus swallows it, and the events
     * it was handling reach no record.
     */
    @Test
    void eachExceptionASubscriberThrowsIsCounted() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        String key = "subscriber HookFailuresTest$ThrowingSubscriber.onPhase";
        CollectionCaps noForks = TestCaps.caps().interventions(0).probes(0).build();
        ShardRun.recordedWithClauseHook(tempDir, game, noForks, "to record a game", () -> {
            long before = HookFailures.counts().getOrDefault(key, 0L);
            game.subscribeToEvents(new ThrowingSubscriber());
            for (int phase = 0; phase < 3; phase++) {
                TestBoard.nextStep(game);
            }
            assertEquals(before + 3, HookFailures.counts().get(key), HookFailures.counts().toString());
        });
    }

    /** Guava's report of a failure names the subscriber's class and method. */
    @Test
    void theCountKeyNamesTheSubscribersClassAndMethod() {
        assertEquals("subscriber BusBracketCollector.onCast", HookFailures.subscriberOf(
                "Exception thrown by subscriber method onCast(forge.game.event."
                        + "GameEventSpellAbilityCast) on subscriber com.pricepredictor.connector."
                        + "effects.BusBracketCollector@1b2c3d when dispatching event: cast"));
        assertEquals("subscriber ?", HookFailures.subscriberOf("something else"));
    }

    /**
     * A hook call with fewer arguments than its callback reads is answered
     * with nothing, its body never run, and counted.
     */
    @Test
    void aHookCallShortOfArgumentsIsCounted() throws Throwable {
        ListenerSetter hook = PatchHooks.PRIORITY_SETTLED_LISTENER;
        Method settled = HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");
        String key = hook + " " + settled.getName() + " short of arguments";
        long before = HookFailures.counts().getOrDefault(key, 0L);
        Listener listener = hook.listener((callback, args) -> {
            throw new AssertionError("a short call reached the body");
        });

        assertNull(listener.invoke(null, settled, new Object[]{}));
        assertNull(listener.invoke(null, settled, null));

        assertEquals(before + 2, HookFailures.counts().get(key), HookFailures.counts().toString());
    }
}
