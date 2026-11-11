package com.pricepredictor.connector.effects;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * How many times each effect-record hook's listener has thrown in this JVM
 * (C11).
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

    /** The failures counted so far, by hook, in hook order; empty when none threw. */
    public static Map<String, Long> counts() {
        Map<String, Long> counts = new TreeMap<>();
        COUNTS.forEach((hook, failures) -> counts.put(hook, failures.get()));
        return counts;
    }
}
