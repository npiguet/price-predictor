package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;

/** The caps and budgets every collecting supervisor shares (FR-028). */
public record CollectionCaps(
        int manaCap,
        double playabilityRate,
        int interventionsPerGame,
        int probesPerGame,
        List<String> probeKeywords,
        double randomSeatShare,
        double randomSeatProbability) {

    /** The gen-1 caps alone: no random seat, which is the default anyway. */
    public CollectionCaps(
            int manaCap,
            double playabilityRate,
            int interventionsPerGame,
            int probesPerGame,
            List<String> probeKeywords) {
        this(manaCap, playabilityRate, interventionsPerGame, probesPerGame,
                probeKeywords, 0.0, 0.0);
    }

    public static CollectionCaps defaults() {
        return new CollectionCaps(
                1, 0.1, 2, 2, List.of(), 0.0, 0.0);
    }

    /**
     * Read from the {@code effect.*} system properties the supervisor sets.
     *
     * <p>Each falls back to its default rather than failing, because a
     * worker started without instrumentation sets none of them and one
     * started by an older supervisor may set only some.
     */
    public static CollectionCaps fromSystemProperties() {
        CollectionCaps defaults = defaults();
        return new CollectionCaps(
                intProperty("effect.mana.cap", defaults.manaCap()),
                doubleProperty("effect.playability.rate", defaults.playabilityRate()),
                intProperty("effect.interventions.per.game",
                        defaults.interventionsPerGame()),
                intProperty("effect.probes.per.game", defaults.probesPerGame()),
                listProperty("effect.probe.keywords"),
                // The random seat's two knobs (FR-021). A share of matches
                // and a per-decision P: both are drawn in the JVM, so the
                // supervisor can only name them. The Python side refuses a
                // share without a P before any worker starts (FR-026); a
                // worker reads what it is given and defaults the rest.
                doubleProperty("effect.random.seat.share", defaults.randomSeatShare()),
                doubleProperty("effect.random.seat.probability",
                        defaults.randomSeatProbability()));
    }

    /** Whether any match of this run may seat a random player at all. */
    public boolean randomSeatEnabled() {
        return randomSeatShare > 0.0;
    }

    private static int intProperty(String name, int fallback) {
        try {
            String value = System.getProperty(name);
            return value == null || value.isBlank()
                    ? fallback : Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static double doubleProperty(String name, double fallback) {
        try {
            String value = System.getProperty(name);
            return value == null || value.isBlank()
                    ? fallback : Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static List<String> listProperty(String name) {
        String value = System.getProperty(name);
        if (value == null || value.isBlank()) {
            return List.of();
        }
        List<String> parsed = new ArrayList<>();
        for (String part : value.split(",")) {
            if (!part.isBlank()) {
                parsed.add(part.trim());
            }
        }
        return List.copyOf(parsed);
    }

    /**
     * Whether any probe fork should be taken at all.
     *
     * <p>With no keywords named, none is — whatever the build state. The
     * flag is the runtime switch, separate from the build decision gate 2
     * drives.
     */
    public boolean probesEnabled() {
        return !probeKeywords.isEmpty();
    }
}
