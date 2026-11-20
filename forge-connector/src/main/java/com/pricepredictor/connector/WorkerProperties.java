package com.pricepredictor.connector;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

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

    /** What a value that does not parse, or fails its constraint, does. */
    public enum OnMalformed {
        /** Name the property on stderr and exit with {@link ExitCode#BAD_CONFIG}. */
        EXIT,
        /** Name the property and the bad value on one stderr line; use the default. */
        WARN_AND_DEFAULT
    }

    private WorkerProperties() {
    }

    /** The property's value, or exit naming it when it is unset or blank. */
    static String require(String name) {
        return optional(name).orElseThrow(
                () -> ToolMain.badConfig("-D" + name + " system property is required"));
    }

    /**
     * The property's value trimmed, or empty when it is unset or blank.
     *
     * <p>For an optional property with no default to fall back on — a file,
     * a set code, a label — whose absence means "not asked for". A blank value
     * is a launcher passing {@code -Dname=} for an option it was not given,
     * and read as given it would name the empty path or the set called "".
     */
    static Optional<String> optional(String name) {
        String value = System.getProperty(name);
        return value == null || value.isBlank() ? Optional.empty() : Optional.of(value.trim());
    }

    /** {@link #optional} as a path. */
    static Optional<Path> optionalPath(String name) {
        return optional(name).map(Path::of);
    }

    /** A comma-separated property as trimmed, non-empty items; empty when unset. */
    public static List<String> listProperty(String name) {
        return csv(System.getProperty(name));
    }

    /**
     * A comma-separated property whose presence is itself the fact: empty when
     * unset, and the empty list when set blank.
     *
     * <p>The one exception to a blank value reading as unset, for a property
     * that says what a run is as well as what it holds — a coverage run aimed
     * at no rule in particular is still aimed, and says so by passing the
     * property with nothing in it.
     */
    public static Optional<List<String>> presentListProperty(String name) {
        String value = System.getProperty(name);
        return value == null ? Optional.empty() : Optional.of(csv(value));
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
    public static boolean flag(String name) {
        return Boolean.parseBoolean(System.getProperty(name, "false"));
    }
}
