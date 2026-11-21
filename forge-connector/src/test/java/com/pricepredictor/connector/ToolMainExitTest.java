package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.OptionalInt;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ToolMain#exit} throws rather than ending the JVM, so every exit path
 * is testable; a {@code main} that does not run under {@link ToolMain#guard}
 * or {@link ToolMain#run} would end an exit it reaches as an uncaught error
 * with status 1 instead of its code.
 */
class ToolMainExitTest {

    @Test
    void everyMainRunsUnderTheExitGuard() throws IOException {
        Path sources = Path.of("src/main/java");
        List<Path> unguarded;
        try (Stream<Path> files = Files.walk(sources)) {
            unguarded = files.filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> {
                        String text = read(path);
                        return text.contains("public static void main(")
                                && !text.contains("ToolMain.guard(")
                                && !text.contains("ToolMain.run(");
                    })
                    .toList();
        }
        assertEquals(List.of(), unguarded);
    }

    /** Each main that reads {@code --flag value} arguments, as its testable body. */
    static Stream<Arguments> toolMains() {
        return Stream.of(
                Arguments.of("CastabilityMain", (Body) CastabilityMain::work),
                Arguments.of("ConvertMain", (Body) ConvertMain::work),
                Arguments.of("KeywordDefinitionMain", (Body) KeywordDefinitionMain::work),
                Arguments.of("PoolMain", (Body) PoolMain::work),
                Arguments.of("VariantSidecarMain", (Body) VariantSidecarMain::work));
    }

    /** A main's body, called with its command line. */
    @FunctionalInterface
    interface Body {
        void run(String[] args) throws Exception;
    }

    /**
     * Every main refuses a flag it does not declare, before any work, and
     * reads its arguments under the guard, so the refusal is its exit
     * status rather than an uncaught error.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("toolMains")
    void everyToolMainRefusesAnUnknownFlag(String name, Body body) {
        assertEquals(ExitCode.BAD_CONFIG, exitOf(body, "--no-such-flag", "x"), name);
    }

    /** No main writes to a default output path: a missing one is refused. */
    @Test
    void theConverterAndTheVariantSidecarWriterNeedTheirOutputPaths() {
        assertEquals(ExitCode.BAD_CONFIG, exitOf(ConvertMain::work,
                "--cards-path", "cards", "--tokens-output-path", "tokens"));
        assertEquals(ExitCode.BAD_CONFIG, exitOf(ConvertMain::work,
                "--cards-path", "cards", "--output-path", "out"));
        assertEquals(ExitCode.BAD_CONFIG, exitOf(VariantSidecarMain::work));
    }

    /** The guard ends the process non-zero on any throwable, not only its own exit. */
    @Test
    void theGuardExitsNonZeroOnAnyThrowable() {
        assertEquals(OptionalInt.of(ExitCode.FAILURE.status()), guardedQuietly(() -> {
            throw new IllegalStateException("holdout property unreadable");
        }));
        assertEquals(OptionalInt.of(ExitCode.FAILURE.status()), guardedQuietly(() -> {
            throw new OutOfMemoryError("Java heap space");
        }));
        assertEquals(OptionalInt.of(ExitCode.BAD_CONFIG.status()), guardedQuietly(() -> {
            throw ToolMain.exit(ExitCode.BAD_CONFIG);
        }));
        assertEquals(OptionalInt.empty(), guardedQuietly(() -> { }));
    }

    @Test
    void theGuardPrintsTheThrowableItEnds() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true, StandardCharsets.UTF_8));
        try {
            ToolMain.guardedStatus(() -> {
                throw new IllegalStateException("holdout property unreadable");
            });
        } finally {
            System.setErr(original);
        }
        assertTrue(captured.toString(StandardCharsets.UTF_8)
                .contains("holdout property unreadable"));
    }

    /**
     * The guard's first line is {@code Fatal error: } and the throwable, the
     * line {@code worker_failures.JVM_ERROR} counts (finding L1-1): a worker
     * linked against a stale Forge jar ends on a {@code NoSuchMethodError}.
     */
    @Test
    void theGuardsFirstLineIsFatalErrorAndTheThrowable() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true, StandardCharsets.UTF_8));
        try {
            ToolMain.guardedStatus(() -> {
                throw new NoSuchMethodError("'void forge.game.Game.foo()'");
            });
        } finally {
            System.setErr(original);
        }
        assertEquals("Fatal error: java.lang.NoSuchMethodError: 'void forge.game.Game.foo()'",
                captured.toString(StandardCharsets.UTF_8).lines().findFirst().orElse(""));
    }

    private static OptionalInt guardedQuietly(Runnable main) {
        PrintStream original = System.err;
        System.setErr(new PrintStream(OutputStream.nullOutputStream()));
        try {
            return ToolMain.guardedStatus(main);
        } finally {
            System.setErr(original);
        }
    }

    /** The exit {@code body} requests on {@code args}, its stderr discarded. */
    private static ExitCode exitOf(Body body, String... args) {
        PrintStream original = System.err;
        System.setErr(new PrintStream(OutputStream.nullOutputStream()));
        try {
            Executable call = () -> body.run(args);
            return assertThrows(ToolMain.Exit.class, call).code();
        } finally {
            System.setErr(original);
        }
    }

    private static String read(Path path) {
        try {
            return Files.readString(path);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
