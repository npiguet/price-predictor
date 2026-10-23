package com.pricepredictor.connector;

import java.util.Map;

/** What the draft worker itself knows about an agent label. */
final class DraftAgent {

    /**
     * Share of picks replaced by a uniform-random legal card. {@code forge-full}
     * and any label not listed here are pure Forge AI.
     */
    private static final Map<String, Double> RANDOM_OVERRIDE_FRACTION =
            Map.of("forge-r30", 0.30, "forge-r100", 1.0);

    private DraftAgent() {
    }

    static double randomOverrideFraction(String label) {
        return RANDOM_OVERRIDE_FRACTION.getOrDefault(label, 0.0);
    }
}
