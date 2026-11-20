package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@link ToolMain.Args}: the {@code --flag value} pairs a tool main reads, and the exit codes it ends on. */
class ToolMainArgsTest {

    private static final ToolMain.Flag CARDS = new ToolMain.Flag("--cards");
    private static final ToolMain.Flag OUTPUT = new ToolMain.Flag("--output");
    private static final ToolMain.Flag SIZE = new ToolMain.Flag("--size");
    private static final ToolMain.Flag REPLACE = new ToolMain.Flag("--replace");
    private static final ToolMain.Flag OUTPUT_PATH = new ToolMain.Flag("--output-path");

    private static final ToolMain.Args ARGS = ToolMain.args(
            new String[] {"--cards", "in.txt", "--output", "out.json"}, Set.of(),
            CARDS, OUTPUT, SIZE);

    /** What a refused command line printed: the problem, on stderr, before the exit. */
    private static String refusal(Executable parse) {
        PrintStream original = System.err;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        System.setErr(new PrintStream(bytes, true, StandardCharsets.UTF_8));
        try {
            ToolMain.Exit exit = assertThrows(ToolMain.Exit.class, parse);
            assertEquals(ExitCode.BAD_CONFIG, exit.code());
        } finally {
            System.setErr(original);
        }
        return bytes.toString(StandardCharsets.UTF_8);
    }

    @Test
    void aGivenFlagIsRead() {
        assertEquals("in.txt", ARGS.require(CARDS));
        assertEquals("out.json", ARGS.require(OUTPUT));
    }

    @Test
    void aSwitchTakesNoValue() {
        ToolMain.Args args = ToolMain.args(
                new String[] {"--replace", "--output", "out"}, Set.of(REPLACE), OUTPUT);

        assertTrue(args.has(REPLACE));
        assertEquals("out", args.require(OUTPUT));
        assertFalse(ARGS.has(REPLACE));
    }

    /**
     * A misspelt flag would otherwise leave its value to the flag's default:
     * a misspelt output path wrote a whole conversion into the working
     * directory.
     */
    @Test
    void anUnknownFlagIsRefusedByName() {
        String printed = refusal(() -> ToolMain.args(
                new String[] {"--ouptut-path", "out"}, Set.of(REPLACE), OUTPUT_PATH));

        assertTrue(printed.contains("--ouptut-path"), printed);
        assertTrue(printed.contains("--output-path"), printed);
    }

    @Test
    void aStrayArgumentIsRefused() {
        String printed = refusal(() -> ToolMain.args(
                new String[] {"--output", "out", "extra"}, Set.of(), OUTPUT));

        assertTrue(printed.contains("extra"), printed);
    }

    @Test
    void aFlagWithoutItsValueIsRefused() {
        String printed = refusal(() -> ToolMain.args(
                new String[] {"--output"}, Set.of(), OUTPUT));

        assertTrue(printed.contains("--output"), printed);
    }

    @Test
    void aMissingRequiredFlagIsNamed() {
        String printed = refusal(() -> ARGS.require(SIZE));

        assertTrue(printed.contains("--size"), printed);
        assertNull(ARGS.get(SIZE));
    }

    @Test
    void aMalformedIntegerIsNamed() {
        ToolMain.Args args = ToolMain.args(new String[] {"--size", "ten"}, Set.of(), SIZE);

        String printed = refusal(() -> args.intValue(SIZE, 1));

        assertTrue(printed.contains("--size") && printed.contains("ten"), printed);
    }

    /**
     * The supervisors test only for non-zero, but the codes are what an
     * operator reads, so they are pinned: a configuration error is told apart
     * from a failed run.
     */
    @Test
    void theExitCodesAreTheDocumentedValues() {
        assertEquals(0, ExitCode.OK.status());
        assertEquals(1, ExitCode.FAILURE.status());
        assertEquals(2, ExitCode.BAD_CONFIG.status());
        assertEquals(ExitCode.BAD_CONFIG.status(), ExitCode.IO_FAILURE.status());
    }
}
