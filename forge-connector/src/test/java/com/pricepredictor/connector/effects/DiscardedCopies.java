package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.StdStreams;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A fork whose game copy fails on purpose, run so that the discard is
 * asserted and its first-failure line stays off the test output.
 */
final class DiscardedCopies {

    private DiscardedCopies() {
    }

    /**
     * Run {@code body}, which takes one fork of {@code kind} over a game that
     * cannot be copied, and assert that exactly one copy failure was counted
     * for that kind. The line the first failure prints is captured, not shown.
     */
    static void expectOne(String kind, StdStreams.Body body) throws Throwable {
        String key = "fork " + kind + " copy";
        long before = HookFailures.counts().getOrDefault(key, 0L);
        StdStreams.captureStderr(body);
        assertEquals(before + 1, HookFailures.counts().getOrDefault(key, 0L), key + " counted once");
    }
}
