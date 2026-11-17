package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
