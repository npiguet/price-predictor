package com.pricepredictor.connector;

import com.pricepredictor.connector.WorkerProperties.OnMalformed;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Predicate;

/**
 * The numeric {@code -D} system properties the worker mains are configured by,
 * each read as its {@link IntSetting} or {@link DoubleSetting} says. What a
 * malformed value does is the caller's {@link OnMalformed} choice
 * ({@link WorkerProperties} says why the two kinds differ); a blank value
 * counts as unset under either policy.
 */
public final class NumericProperties {

    /**
     * What an integer property's value must satisfy, with the sentence a
     * refusal gives for it ({@code "must be >= 1"}).
     */
    public record IntConstraint(IntPredicate isValid, String demand) {
        /** No constraint beyond parsing. */
        static final IntConstraint NONE = new IntConstraint(v -> true, "");

        /** At least {@code minimum}. */
        public static IntConstraint atLeast(int minimum) {
            return new IntConstraint(v -> v >= minimum, "must be >= " + minimum);
        }
    }

    /** An integer property: its name, the value when unset, and what a set value must satisfy. */
    public record IntSetting(String name, int defaultValue, IntConstraint constraint) {
        /** A setting with no constraint beyond parsing. */
        public static IntSetting of(String name, int defaultValue) {
            return new IntSetting(name, defaultValue, IntConstraint.NONE);
        }

        /**
         * Games per match. Odd because Forge plays first to {@code N/2 + 1}
         * wins: an even {@code N} can run past {@code N} games, and a match
         * row's winner sequence is promised to be at most {@code N} long.
         */
        public static IntSetting bestOf(String name, int defaultValue) {
            return new IntSetting(name, defaultValue, new IntConstraint(
                    NumericProperties::isValidBestOf, "must be a positive odd integer"));
        }
    }

    /** A decimal property: its name and the value when unset. */
    public record DoubleSetting(String name, double defaultValue) {
    }

    /**
     * How a property's text becomes a {@code T}: the parse, what the refusal
     * says was expected when it does not parse, and what the parsed value
     * must satisfy.
     */
    private record Format<T>(
            Function<String, T> parse, String expected, Predicate<T> isValid, String demand) {
    }

    private NumericProperties() {
    }

    /**
     * An integer property, its default when unset; a value that does not
     * parse or fails its constraint is handled per {@code policy}.
     */
    public static int intProperty(IntSetting setting, OnMalformed policy) {
        IntConstraint constraint = setting.constraint();
        return read(setting.name(), setting.defaultValue(), policy, new Format<>(
                Integer::valueOf, "an integer", constraint.isValid()::test, constraint.demand()));
    }

    /** A decimal property with no constraint beyond parsing. */
    public static double doubleProperty(DoubleSetting setting, OnMalformed policy) {
        return read(setting.name(), setting.defaultValue(), policy,
                new Format<>(Double::valueOf, "a number", v -> true, ""));
    }

    /**
     * A probability, in [0, 1]; empty when unset or blank. Anything else,
     * a value that does not parse included, is refused as
     * {@link OnMalformed#EXIT} refuses: there is no default to fall back on.
     */
    public static Optional<Double> probability(String name) {
        return Optional.ofNullable(read(name, (Double) null, OnMalformed.EXIT, new Format<>(
                Double::valueOf, "a number", value -> value >= 0.0 && value <= 1.0,
                "must be in [0, 1]")));
    }

    /** The games-per-match property ({@link IntSetting#bestOf}), refusing as {@link OnMalformed#EXIT} does. */
    public static int bestOf(IntSetting setting) {
        return intProperty(setting, OnMalformed.EXIT);
    }

    static boolean isValidBestOf(int gamesPerMatch) {
        return gamesPerMatch >= 1 && gamesPerMatch % 2 == 1;
    }

    private static <T> T read(String name, T defaultValue, OnMalformed policy, Format<T> format) {
        String raw = System.getProperty(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        T value;
        try {
            value = format.parse().apply(raw.trim());
        } catch (NumberFormatException e) {
            return malformed(name, "must be " + format.expected() + ", got: " + raw, defaultValue, policy);
        }
        if (!format.isValid().test(value)) {
            return malformed(name, format.demand() + ", got: " + value, defaultValue, policy);
        }
        return value;
    }

    private static <T> T malformed(String name, String problem, T defaultValue, OnMalformed policy) {
        if (policy == OnMalformed.EXIT) {
            throw ToolMain.badConfig("-D" + name + " " + problem);
        }
        System.err.println("Warning: -D" + name + " " + problem + "; using default " + defaultValue);
        return defaultValue;
    }
}
