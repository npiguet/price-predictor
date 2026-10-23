package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@link ToolMain.Args}: the {@code --flag value} pairs a tool main reads, and the exit codes it ends on. */
class ToolMainArgsTest {

    private static final ToolMain.Args ARGS = ToolMain.args(
            new String[] {"--cards", "in.txt", "--stale", "--output", "out.json"},
            "--cards", "--output", "--size");

    @Test
    void aGivenFlagIsReadAndAnUnknownOneSwallowsNothing() {
        assertEquals("in.txt", ARGS.require("--cards"));
        assertEquals("out.json", ARGS.require("--output"));
    }

    @Test
    void aMissingRequiredFlagIsNamed() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> ARGS.require("--size"));
        assertTrue(e.getMessage().contains("--size"), e.getMessage());
        assertNull(ARGS.get("--size"));
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
