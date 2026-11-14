package com.pricepredictor.connector.effects;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * How many times each effect-record hook's listener has thrown in this JVM
 * (C11), how many calls reached one short of the arguments its callback
 * reads, and how many times a collector's bus subscriber threw.
 *
 * <p>Forge contains a listener's failure ({@code EffectRecordHook.contain}):
 * the engine's call goes on, the slot's first failure is reported once, and
 * the listener <b>keeps being called</b>. So a failing hook does not go
 * quiet; each call that throws loses what that call would have recorded, and
 * only a count says how many were lost. Every listener this side installs is
 * a {@link PatchHooks.Listener}, which counts here what its body throws
 * before letting it go on to Forge.
 *
 * <p>The first failure of each hook is printed with its stack, which Forge's
 * own one-line report leaves out.
 */
public final class HookFailures {

    /** Stack frames printed with a hook's first failure. */
    private static final int FRAMES_SHOWN = 25;

    private static final Map<String, AtomicLong> COUNTS = new ConcurrentHashMap<>();

    /**
     * The logger Guava's {@code EventBus} reports a throwing subscriber to:
     * Forge's bus is {@code new EventBus("game events")}, whose default
     * exception handler logs each failure under the bus class's name and the
     * bus's identifier. Held here so the handler counting on it outlives any
     * collection of the logger.
     */
    private static final Logger GAME_BUS =
            Logger.getLogger("com.google.common.eventbus.EventBus.game events");

    /** Whether {@link #watchSubscribers} has put its handler on {@link #GAME_BUS}. */
    private static final AtomicBoolean WATCHING = new AtomicBoolean();

    /** A subscriber method and its class in Guava's report of a failure. */
    private static final Pattern SUBSCRIBER =
            Pattern.compile("subscriber method (\\w+)\\(.*?\\) on subscriber ([\\w.$]+)@");

    private HookFailures() {
    }

    /** Count one call of {@code hook}'s listener that threw {@code failure}. */
    static void count(PatchHooks.Hook hook, RuntimeException failure) {
        AtomicLong failures = COUNTS.computeIfAbsent(hook.toString(), name -> new AtomicLong());
        if (failures.incrementAndGet() == 1) {
            System.err.println("effect-record hook " + hook + ": its listener threw, which loses what"
                    + " that call would have recorded (later failures are counted, not printed): "
                    + failure + Arrays.stream(failure.getStackTrace()).limit(FRAMES_SHOWN)
                    .map(frame -> "\n\tat " + frame).collect(Collectors.joining()));
        }
    }

    /**
     * Count one call of {@code hook}'s listener that brought fewer arguments
     * than {@code method}'s callback reads: a Forge whose listener lost an
     * argument the collector reads. The call is answered with nothing, the
     * way an absent hook's would be, and loses what it would have recorded;
     * counted under {@code <hook> <method> short of arguments}, the first
     * printed.
     */
    static void countShortCall(PatchHooks.Hook hook, String method, int given, int read) {
        String key = hook + " " + method + " short of arguments";
        AtomicLong calls = COUNTS.computeIfAbsent(key, name -> new AtomicLong());
        if (calls.incrementAndGet() == 1) {
            System.err.println("effect-record hook " + hook + ": " + method + " was called with "
                    + given + " argument(s) and its listener reads " + read + "; the call is"
                    + " ignored, which loses what it would have recorded (later ones are counted,"
                    + " not printed)");
        }
    }

    /**
     * Count every exception a subscriber on a game's bus throws, from now on
     * in this JVM: the collectors' {@code @Subscribe} methods, a fork's
     * included. Guava's {@code EventBus} logs such an exception and swallows
     * it, leaving the collector as far as it got, so only a count says how
     * many events were lost. Counted under {@code subscriber <class>.<method>}.
     * Once per JVM; a later call does nothing.
     */
    static void watchSubscribers() {
        if (!WATCHING.compareAndSet(false, true)) {
            return;
        }
        GAME_BUS.addHandler(new Handler() {
            @Override
            public void publish(LogRecord record) {
                if (record != null && record.getThrown() != null) {
                    COUNTS.computeIfAbsent(subscriberOf(record.getMessage()),
                            name -> new AtomicLong()).incrementAndGet();
                }
            }

            @Override
            public void flush() {
            }

            @Override
            public void close() {
            }
        });
    }

    /** The count key of the subscriber Guava's failure message names. */
    static String subscriberOf(String message) {
        Matcher named = message == null ? null : SUBSCRIBER.matcher(message);
        if (named == null || !named.find()) {
            return "subscriber ?";
        }
        String type = named.group(2);
        return "subscriber " + type.substring(type.lastIndexOf('.') + 1) + "." + named.group(1);
    }

    /**
     * A failure the connector contains outside a hook's listener and a bus
     * subscriber, counted under {@code key}: a phase boundary's step
     * ({@code "boundary step <name>"}), a decision candidate the rules check
     * could not read ({@code "decision candidate unreadable"}).
     */
    static void countContained(String key) {
        COUNTS.computeIfAbsent(key, name -> new AtomicLong()).incrementAndGet();
    }

    /**
     * The same, the first failure under {@code key} printed with its stack:
     * a fork whose copy or whose body threw ({@code "fork <kind> copy"},
     * {@code "fork <kind> body"}), which is discarded rather than failing the
     * worker, and which would otherwise leave no trace.
     */
    static void countContained(String key, Throwable failure) {
        if (COUNTS.computeIfAbsent(key, name -> new AtomicLong()).incrementAndGet() == 1) {
            System.err.println("Effect records: " + key + " threw, and is discarded (later failures"
                    + " are counted, not printed): " + failure
                    + Arrays.stream(failure.getStackTrace()).limit(FRAMES_SHOWN)
                    .map(frame -> "\n\tat " + frame).collect(Collectors.joining()));
        }
    }

    /**
     * The failures counted so far, by hook, short call, subscriber or
     * contained failure, in name order; empty when none failed.
     */
    public static Map<String, Long> counts() {
        Map<String, Long> counts = new TreeMap<>();
        COUNTS.forEach((hook, failures) -> counts.put(hook, failures.get()));
        return counts;
    }
}
