package com.pricepredictor.connector;

import java.util.List;
import java.util.Random;

/**
 * The set each pool or draft is opened from: the one the caller named, or a
 * fresh uniform draw over the sealed-legal sets every time.
 *
 * <p>The eligible sets are computed once, at construction, and only when no
 * set is named; the random source is touched only by a draw, so a named set
 * leaves its stream exactly as it was.
 */
final class SetChooser {

    private final String namedSet;
    private final List<String> eligibleSets;
    private final Random random;

    private SetChooser(String namedSet, List<String> eligibleSets, Random random) {
        this.namedSet = namedSet;
        this.eligibleSets = eligibleSets;
        this.random = random;
    }

    /**
     * A chooser for {@code namedSet}, or for a random eligible set per call
     * when it is null.
     *
     * @throws IllegalStateException when no set is named and none is eligible
     */
    static SetChooser of(String namedSet, Random random) {
        if (namedSet != null) {
            return new SetChooser(namedSet, List.of(), random);
        }
        List<String> eligible = MatchGenerator.computeEligibleSets();
        if (eligible.isEmpty()) {
            throw new IllegalStateException("no eligible sealed-legal sets found");
        }
        return new SetChooser(null, eligible, random);
    }

    /** True when every call draws a set rather than returning the named one. */
    boolean isRandom() {
        return namedSet == null;
    }

    /** How many sets a draw chooses among; zero for a named set. */
    int eligibleCount() {
        return eligibleSets.size();
    }

    /** The named set, or one eligible set drawn uniformly. */
    String next() {
        return namedSet != null
                ? namedSet
                : eligibleSets.get(random.nextInt(eligibleSets.size()));
    }
}
