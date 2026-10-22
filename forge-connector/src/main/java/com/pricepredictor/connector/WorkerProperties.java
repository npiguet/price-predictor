package com.pricepredictor.connector;

import java.util.function.IntPredicate;

/**
 * The {@code -D} system properties the worker mains are configured by.
 *
 * <p>A bad value exits the JVM with {@link #BAD_CONFIG_EXIT} rather than
 * falling back to a default: a worker that guessed would play a whole run in a
 * format nobody asked for, and the supervisor only sees that the exit code is
 * non-zero, which is what gets the operator's attention.
 */
final class WorkerProperties {

    /** Exit code for a missing or malformed property. */
    static final int BAD_CONFIG_EXIT = 2;

    private WorkerProperties() {
    }

    /** The property's value, or exit naming it when it is unset or blank. */
    static String require(String name) {
        String value = System.getProperty(name);
        if (value == null || value.isBlank()) {
            System.err.println("Error: -D" + name + " system property is required");
            System.exit(BAD_CONFIG_EXIT);
        }
        return value;
    }

    /**
     * An integer property, {@code defaultValue} when unset; exits on a value
     * that does not parse or fails {@code isValid}.
     *
     * @param constraintMsg what {@code isValid} demands, e.g. {@code "must be >= 1"}
     */
    static int intProperty(
            String name, int defaultValue, IntPredicate isValid, String constraintMsg) {
        String prop = System.getProperty(name);
        if (prop == null) {
            return defaultValue;
        }
        int value;
        try {
            value = Integer.parseInt(prop);
        } catch (NumberFormatException e) {
            System.err.println("Error: -D" + name + " must be an integer, got: " + prop);
            System.exit(BAD_CONFIG_EXIT);
            return -1;  // unreachable
        }
        if (!isValid.test(value)) {
            System.err.println("Error: -D" + name + " " + constraintMsg + ", got: " + value);
            System.exit(BAD_CONFIG_EXIT);
        }
        return value;
    }

    /**
     * Games per match. Odd because Forge plays first to {@code N/2 + 1} wins:
     * an even {@code N} can run past {@code N} games, and a match row's
     * winner sequence is promised to be at most {@code N} long.
     */
    static int bestOf(String name, int defaultValue) {
        return intProperty(name, defaultValue, WorkerProperties::isValidBestOf,
                "must be a positive odd integer");
    }

    static boolean isValidBestOf(int gamesPerMatch) {
        return gamesPerMatch >= 1 && gamesPerMatch % 2 == 1;
    }

    /** A boolean property, false when unset. */
    static boolean flag(String name) {
        return Boolean.parseBoolean(System.getProperty(name, "false"));
    }
}
