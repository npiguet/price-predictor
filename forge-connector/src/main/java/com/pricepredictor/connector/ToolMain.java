package com.pricepredictor.connector;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * What the one-shot tool mains share: {@code --flag value} arguments, and the
 * exit protocol the Python side relies on — 0 when the work returns, 1 with a
 * stack trace on stderr when it throws. The callers only check for non-zero,
 * so every failure is one exit code; the message is for the operator.
 */
final class ToolMain {

    /** The work itself, Forge initialization included where it needs one. */
    @FunctionalInterface
    interface Body {
        void run() throws Exception;
    }

    private ToolMain() {
    }

    /** Run {@code body}, then exit 0, or exit 1 if it throws. */
    static void run(Body body) {
        try {
            body.run();
            System.exit(0);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    /**
     * The values of the named {@code --flag value} pairs; the last occurrence
     * wins.
     *
     * <p>Only the named flags take a value. Anything else on the command line
     * is skipped on its own, so a flag this main does not know never swallows
     * the next argument — an older caller passing a since-removed flag still
     * gets the rest of its arguments read.
     */
    static Args args(String[] args, String... flags) {
        Set<String> known = Set.of(flags);
        Map<String, String> values = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (known.contains(args[i]) && i + 1 < args.length) {
                values.put(args[i], args[++i]);
            }
        }
        return new Args(values);
    }

    /** Parsed {@code --flag value} pairs. */
    record Args(Map<String, String> values) {

        /** The flag's value, or {@code defaultValue} when it was not given. */
        String get(String flag, String defaultValue) {
            return values.getOrDefault(flag, defaultValue);
        }

        /** The flag's value, or null when it was not given. */
        String get(String flag) {
            return values.get(flag);
        }
    }
}
