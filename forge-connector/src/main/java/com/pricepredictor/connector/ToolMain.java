package com.pricepredictor.connector;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;

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
        } catch (Exit exit) {
            System.exit(exit.status());
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(ExitCode.FAILURE.status());
        }
    }

    /**
     * Run a worker's {@code main}, ending the process with the code of an
     * {@link Exit} thrown anywhere under it, or {@link ExitCode#FAILURE} with
     * the throwable on stderr for anything else, an {@link Error} out of a
     * game included; a {@code main} that returns ends as it would have
     * unguarded.
     *
     * <p>Every main that reaches {@link #exit} or {@link #badConfig} runs
     * under this or {@link #run}: those throw rather than call
     * {@link System#exit} themselves, so an exit path can be tested by
     * catching the {@link Exit} it throws.
     */
    static void guard(Runnable main) {
        OptionalInt status = guardedStatus(main);
        if (status.isPresent()) {
            System.out.flush();
            System.err.flush();
            System.exit(status.getAsInt());
        }
    }

    /**
     * The exit status {@link #guard} ends the process with for {@code main},
     * empty when it returns. A throwable that is not an {@link Exit} is
     * printed first: left uncaught it would end only the main thread, and
     * the JVM would stay up until Forge's non-daemon pool threads idled out,
     * holding the worker's slot.
     */
    static OptionalInt guardedStatus(Runnable main) {
        try {
            main.run();
            return OptionalInt.empty();
        } catch (Exit exit) {
            return OptionalInt.of(exit.status());
        } catch (Throwable thrown) {
            System.err.println("Fatal error: " + thrown);
            thrown.printStackTrace(System.err);
            return OptionalInt.of(ExitCode.FAILURE.status());
        }
    }

    /**
     * The process is to end with {@code code}. Returned for the caller to
     * {@code throw}, so the compiler sees that control does not continue;
     * {@link #run} or {@link #guard} turns it into the exit status.
     */
    static Exit exit(ExitCode code) {
        return new Exit(code);
    }

    /**
     * Name the problem on stderr; the result, for the caller to
     * {@code throw}, ends the process with {@link ExitCode#BAD_CONFIG}, as in
     * {@link #exit}.
     */
    static Exit badConfig(String problem) {
        System.err.println("Error: " + problem);
        return exit(ExitCode.BAD_CONFIG);
    }

    /**
     * A request to end the process with an exit code.
     *
     * <p>An {@link Error} rather than an exception, so no {@code catch
     * (Exception)} on its way to {@link #guard} — a worker loop's, a tool
     * main's — mistakes it for a failure of the work.
     */
    static final class Exit extends Error {
        private final ExitCode code;

        Exit(ExitCode code) {
            super("exit(" + code + ")", null, false, false);
            this.code = code;
        }

        ExitCode code() {
            return code;
        }

        int status() {
            return code.status();
        }
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
     * The values of the named {@code --flag value} pairs, the last occurrence
     * winning, and which of the {@code switches}, flags that take no value,
     * were given ({@link Args#has}).
     *
     * <p>Anything else on the command line is refused, as is a flag left
     * without its value: a misspelt flag would otherwise leave its value to
     * the main's default, and a caller passing a flag the main ignores
     * believes it was honoured.
     *
     * @throws Exit with {@link ExitCode#BAD_CONFIG}, the argument named
     */
    static Args args(String[] args, Set<String> switches, String... flags) {
        Set<String> known = Set.of(flags);
        Map<String, String> values = new HashMap<>();
        Set<String> given = new HashSet<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (switches.contains(arg)) {
                given.add(arg);
            } else if (!known.contains(arg)) {
                throw badConfig("unknown argument " + arg + "; this tool takes "
                        + new TreeSet<>(union(known, switches)));
            } else if (i + 1 == args.length) {
                throw badConfig(arg + " needs a value");
            } else {
                values.put(arg, args[++i]);
            }
        }
        return new Args(values, given);
    }

    private static Set<String> union(Set<String> first, Set<String> second) {
        Set<String> all = new HashSet<>(first);
        all.addAll(second);
        return all;
    }

    /** Parsed {@code --flag value} pairs, and the switches given. */
    record Args(Map<String, String> values, Set<String> switches) {

        /** Whether the switch {@code flag} was given. */
        boolean has(String flag) {
            return switches.contains(flag);
        }

        /** The flag's value, or {@code defaultValue} when it was not given. */
        String get(String flag, String defaultValue) {
            return values.getOrDefault(flag, defaultValue);
        }

        /**
         * The flag's value as an integer, or {@code defaultValue} when it was
         * not given.
         *
         * @throws Exit with {@link ExitCode#BAD_CONFIG} when the value is not an integer
         */
        int intValue(String flag, int defaultValue) {
            String value = values.get(flag);
            if (value == null) {
                return defaultValue;
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                throw badConfig(flag + " must be an integer, got: " + value);
            }
        }

        /** The flag's value, or null when it was not given. */
        String get(String flag) {
            return values.get(flag);
        }

        /**
         * The flag's value.
         *
         * @throws Exit with {@link ExitCode#BAD_CONFIG}, the flag named, when it
         *         was not given
         */
        String require(String flag) {
            String value = values.get(flag);
            if (value == null) {
                throw badConfig(flag + " is required");
            }
            return value;
        }
    }
}
