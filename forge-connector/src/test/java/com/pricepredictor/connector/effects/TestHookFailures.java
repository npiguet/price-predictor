package com.pricepredictor.connector.effects;

import java.util.LinkedHashMap;
import java.util.Map;

/** {@link HookFailures}' counting and the other worker tallies, for a test outside this package. */
public final class TestHookFailures {

    private TestHookFailures() {
    }

    /** Count one contained failure under {@code key}. */
    public static void count(String key) {
        HookFailures.countContained(new HookFailures.FailureKey(key));
    }

    /** Count one fork of {@code kind} that refused, for {@code reason}. */
    public static void refuse(String kind, String reason) {
        ForkRefusals.count(kind, reason);
    }

    /**
     * Count one more {@code count} (a {@link DirectedForkTally.Count} name,
     * {@code "TAKEN"}) for {@code rule} in the JVM lifetime's directed fork tally.
     */
    public static void tallyDirected(String rule, String count) {
        DirectedForkTally.lifetime().add(rule, DirectedForkTally.Count.valueOf(count));
    }

    /** {@code rule}'s counts in the lifetime tally, by the names the tally's line writes. */
    public static Map<String, Long> directedCounts(String rule) {
        Map<String, Long> counts = new LinkedHashMap<>();
        for (DirectedForkTally.Count count : DirectedForkTally.Count.values()) {
            counts.put(count.wireName(), DirectedForkTally.lifetime().get(rule, count));
        }
        return counts;
    }
}
