package com.pricepredictor.connector.effects;

/** {@link HookFailures}' counting, for a test outside this package. */
public final class TestHookFailures {

    private TestHookFailures() {
    }

    /** Count one contained failure under {@code key}. */
    public static void count(String key) {
        HookFailures.countContained(key);
    }
}
