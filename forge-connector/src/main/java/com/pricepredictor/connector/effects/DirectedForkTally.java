package com.pricepredictor.connector.effects;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * How many directed forks each rule has taken in this worker JVM's lifetime,
 * kept or discarded (FR-010a).
 *
 * <p>When several rules match one boundary, the one with the fewest goes
 * first, so a rule whose precondition is common does not take every game's
 * budget from a rare one. A JVM lifetime, not a run: the supervisor recycles
 * workers, and a worker can count only what it saw. One {@link #lifetime()}
 * per JVM; a test builds its own.
 */
final class DirectedForkTally {

    private static final DirectedForkTally LIFETIME = new DirectedForkTally();

    private final Map<String, Integer> taken = new ConcurrentHashMap<>();

    /** The tally of this JVM's lifetime. */
    static DirectedForkTally lifetime() {
        return LIFETIME;
    }

    /** The directed forks {@code rule} has taken so far. */
    int taken(String rule) {
        return taken.getOrDefault(rule, 0);
    }

    /** Count one more fork taken for {@code rule}. */
    void count(String rule) {
        taken.merge(rule, 1, Integer::sum);
    }
}
