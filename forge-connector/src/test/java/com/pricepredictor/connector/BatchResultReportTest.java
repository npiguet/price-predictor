package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchResultReportTest {

    private static String report(BatchConverter.BatchResult result, String tree) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        result.report(new PrintStream(bytes, true, StandardCharsets.UTF_8), tree);
        return bytes.toString(StandardCharsets.UTF_8).replace("\r\n", "\n");
    }

    @Test
    void reportsTheCountsUnderTheTreeName() {
        String text = report(
                new BatchConverter.BatchResult(5, 5, List.of(), List.of()), "variant-scripts");

        assertEquals("  [variant-scripts]\n  Total files:  5\n  Succeeded:    5\n"
                + "  Warnings:     0\n", text);
    }

    @Test
    void listsEachWarningUnderAHeader() {
        String text = report(
                new BatchConverter.BatchResult(3, 1, List.of("[a] bad", "[b] worse"), List.of()),
                "cardsfolder");

        assertTrue(text.contains("  Warnings:     2\n"), text);
        assertTrue(text.endsWith("\nWarnings:\n  [a] bad\n  [b] worse\n"), text);
        assertFalse(text.contains("Warnings:\n  [a] bad\n  [b] worse\n  [a]"), text);
    }
}
