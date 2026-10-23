package com.pricepredictor.connector;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * What the one-shot tool mains share: {@code --flag value} arguments, and the
 * exit protocol the Python side relies on — {@link ExitCode#OK} when the work
 * returns, {@link ExitCode#FAILURE} with a stack trace on stderr when it
 * throws. The callers only check for non-zero; the message is for the
 * operator.
 *
 * <p>It also owns the stream encoding of the workers whose pipes Python reads:
 * Python decodes them strictly as UTF-8, but the JVM encodes {@code System.out}
 * and {@code System.err} with the platform default (Cp1252 on Windows), so one
 * accented card name such as {@code Éowyn} would abort the supervisor with a
 * {@code UnicodeDecodeError}.
 */
final class ToolMain {

    /** The work itself, Forge initialization included where it needs one. */
    @FunctionalInterface
    interface Body {
        void run() throws Exception;
    }

    private ToolMain() {
    }

    /** Run {@code body}, then exit {@link ExitCode#OK}, or {@link ExitCode#FAILURE} if it throws. */
    static void run(Body body) {
        try {
            body.run();
            System.exit(ExitCode.OK.status());
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(ExitCode.FAILURE.status());
        }
    }

    /**
     * Exit with {@code code}. Declared to return an exception only so a
     * caller can write {@code throw exit(..)} and have the compiler see that
     * control does not continue.
     */
    static IllegalStateException exit(ExitCode code) {
        System.exit(code.status());
        return new IllegalStateException("exit(" + code + ") returned");
    }

    /**
     * Exit with {@link ExitCode#BAD_CONFIG} after naming the problem on
     * stderr; the result is for the caller to {@code throw}, as in
     * {@link #exit}.
     */
    static IllegalStateException badConfig(String problem) {
        System.err.println("Error: " + problem);
        return exit(ExitCode.BAD_CONFIG);
    }

    /**
     * A UTF-8 stream bound to the real stdout descriptor; Forge's incidental
     * logging on {@code System.out} goes to a UTF-8 {@code System.err} instead,
     * so the pipe the supervisor reads carries only the caller's own lines.
     */
    static PrintStream claimStdout() {
        PrintStream protocolOut = utf8(new FileOutputStream(FileDescriptor.out));
        System.setErr(utf8(new FileOutputStream(FileDescriptor.err)));
        System.setOut(System.err);
        return protocolOut;
    }

    static PrintStream utf8(OutputStream sink) {
        return new PrintStream(sink, true, StandardCharsets.UTF_8);
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

        /**
         * The flag's value as an integer, or {@code defaultValue} when it was
         * not given.
         *
         * @throws IllegalArgumentException when the value is not an integer
         */
        int intValue(String flag, int defaultValue) {
            String value = values.get(flag);
            if (value == null) {
                return defaultValue;
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(flag + " must be an integer, got: " + value);
            }
        }

        /** The flag's value, or null when it was not given. */
        String get(String flag) {
            return values.get(flag);
        }

        /**
         * The flag's value.
         *
         * @throws IllegalArgumentException naming the flag when it was not given
         */
        String require(String flag) {
            String value = values.get(flag);
            if (value == null) {
                throw new IllegalArgumentException(flag + " is required");
            }
            return value;
        }
    }
}
