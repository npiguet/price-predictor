package com.pricepredictor.connector.effects;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * How many forks of each kind declined in this JVM for an expected reason,
 * by reason: a tally beside {@link HookFailures}' failure counts, and never
 * one of them.
 *
 * <p>A fork refuses when what it was asked cannot happen on the copy: the
 * live ability it would force has no counterpart there ({@link ForkRefusal#NO_COUNTERPART}),
 * its line has no legal mode set with its targets there ({@link ForkRefusal#NO_CHOICES}),
 * or a board edit cannot apply to the card it is bound to (a keyword the card
 * cannot have, a permanent whose state {@code replace_card} cannot carry). Its
 * budget is spent, as every taken fork's is, and it writes nothing; counted
 * here, a run's forks that wrote nothing are told apart from forks that
 * failed. Keys are {@code <kind> <reason>}; the worker prints the tally at a
 * match end whenever it changed, in its status line ({@code WorkerStatus}).
 */
public final class ForkRefusals {

    private static final Map<String, AtomicLong> COUNTS = new ConcurrentHashMap<>();

    private ForkRefusals() {
    }

    /** Count one fork of {@code kind} that refused, for {@code reason}. */
    static void count(String kind, String reason) {
        COUNTS.computeIfAbsent(kind + " " + reason, key -> new AtomicLong()).incrementAndGet();
    }

    /** The refusals counted so far, by kind and reason, in name order; empty when none. */
    public static Map<String, Long> counts() {
        Map<String, Long> counts = new TreeMap<>();
        COUNTS.forEach((key, refused) -> counts.put(key, refused.get()));
        return counts;
    }
}
