package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Predicate;

/**
 * The {@code -D} system properties the worker mains are configured by.
 *
 * <p>What a malformed value does is the caller's explicit choice, because the
 * two kinds of property differ. A worker's own configuration is
 * {@link OnMalformed#EXIT}: a worker that guessed would play a whole run in a
 * format nobody asked for, and the supervisor only sees that the exit code is
 * non-zero, which is what gets the operator's attention. An optional
 * instrumentation knob is {@link OnMalformed#WARN_AND_DEFAULT}: a worker
 * started by an older supervisor may carry only some of them, and a bad one
 * must not stop the run. A blank value counts as unset under either policy.
 */
public final class WorkerProperties {

    /** Exit code for a missing or malformed property. */
    static final int BAD_CONFIG_EXIT = 2;

    /** What a value that does not parse, or fails its constraint, does. */
    public enum OnMalformed {
        /** Name the property on stderr and exit with {@link #BAD_CONFIG_EXIT}. */
        EXIT,
        /** Name the property and the bad value on one stderr line; use the default. */
        WARN_AND_DEFAULT
    }

    private WorkerProperties() {
    }

    /** The property's value, or exit naming it when it is unset or blank. */
    static String require(String name) {
        String value = System.getProperty(name);
        if (value == null || value.isBlank()) {
            throw badConfig("-D" + name + " system property is required");
        }
        return value;
    }

    /**
     * An integer property, {@code defaultValue} when unset; a value that does
     * not parse or fails {@code isValid} is handled per {@code policy}.
     *
     * @param constraintMsg what {@code isValid} demands, e.g. {@code "must be >= 1"}
     */
    static int intProperty(
            String name, int defaultValue, OnMalformed policy,
            IntPredicate isValid, String constraintMsg) {
        return read(name, defaultValue, policy, Integer::valueOf, "an integer",
                isValid::test, constraintMsg);
    }

    /** An integer property with no constraint beyond parsing. */
    public static int intProperty(String name, int defaultValue, OnMalformed policy) {
        return intProperty(name, defaultValue, policy, v -> true, "");
    }

    /** A decimal property with no constraint beyond parsing. */
    public static double doubleProperty(String name, double defaultValue, OnMalformed policy) {
        return read(name, defaultValue, policy, Double::valueOf, "a number", v -> true, "");
    }

    /**
     * Games per match. Odd because Forge plays first to {@code N/2 + 1} wins:
     * an even {@code N} can run past {@code N} games, and a match row's
     * winner sequence is promised to be at most {@code N} long.
     */
    static int bestOf(String name, int defaultValue) {
        return intProperty(name, defaultValue, OnMalformed.EXIT,
                WorkerProperties::isValidBestOf, "must be a positive odd integer");
    }

    static boolean isValidBestOf(int gamesPerMatch) {
        return gamesPerMatch >= 1 && gamesPerMatch % 2 == 1;
    }

    /** A comma-separated property as trimmed, non-empty items; empty when unset. */
    public static List<String> listProperty(String name) {
        return csv(System.getProperty(name));
    }

    /** The trimmed, non-empty items of a comma-separated value; empty for null or blank. */
    static List<String> csv(String value) {
        if (value == null || value.isBlank()) {
            return List.of();
        }
        List<String> items = new ArrayList<>();
        for (String part : value.split(",")) {
            if (!part.isBlank()) {
                items.add(part.trim());
            }
        }
        return List.copyOf(items);
    }

    /**
     * A boolean property, false when unset. Any value but {@code true} reads
     * as false, so there is no malformed value to apply a policy to.
     */
    static boolean flag(String name) {
        return Boolean.parseBoolean(System.getProperty(name, "false"));
    }

    private static <T> T read(
            String name, T defaultValue, OnMalformed policy, Function<String, T> parse,
            String expected, Predicate<T> isValid, String constraintMsg) {
        String raw = System.getProperty(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        T value;
        try {
            value = parse.apply(raw.trim());
        } catch (NumberFormatException e) {
            return malformed(name, "must be " + expected + ", got: " + raw, defaultValue, policy);
        }
        if (!isValid.test(value)) {
            return malformed(name, constraintMsg + ", got: " + value, defaultValue, policy);
        }
        return value;
    }

    private static <T> T malformed(String name, String problem, T defaultValue, OnMalformed policy) {
        if (policy == OnMalformed.EXIT) {
            throw badConfig("-D" + name + " " + problem);
        }
        System.err.println("Warning: -D" + name + " " + problem + "; using default " + defaultValue);
        return defaultValue;
    }

    /**
     * Exit with {@link #BAD_CONFIG_EXIT} after naming the problem; the result
     * is for the caller to {@code throw}, as in {@link ToolMain#exit}.
     */
    static IllegalStateException badConfig(String problem) {
        System.err.println("Error: " + problem);
        return ToolMain.exit(BAD_CONFIG_EXIT);
    }
}
