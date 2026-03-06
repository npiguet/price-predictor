package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for BatchConverter using fixture card scripts.
 * Run via: mvn test -Pintegration
 */
@Tag("integration")
class BatchConverterTest {

    @TempDir
    Path tempDir;

    private Path getFixtureDir() {
        return Path.of("src/test/resources/cardsfolder");
    }

    @Test
    void batchProcessesAllFixtureFiles() throws IOException {
        BatchConverter converter = new BatchConverter();
        BatchConverter.BatchResult result = converter.convert(getFixtureDir(), tempDir);

        assertEquals(3, result.totalFiles());
        assertTrue(result.succeeded() >= 2, "At least 2 valid files should succeed");
    }

    @Test
    void outputDirectoryMirrorsInput() throws IOException {
        BatchConverter converter = new BatchConverter();
        converter.convert(getFixtureDir(), tempDir);

        assertTrue(Files.exists(tempDir.resolve("t/test_bear.txt")));
        assertTrue(Files.exists(tempDir.resolve("t/test_flyer.txt")));
    }

    @Test
    void malformedFileLogsWarningAndContinues() throws IOException {
        BatchConverter converter = new BatchConverter();
        BatchConverter.BatchResult result = converter.convert(getFixtureDir(), tempDir);

        assertTrue(result.warningCount() >= 1, "Malformed file should produce a warning");
        assertTrue(result.succeeded() >= 2, "Valid files should still be converted");
    }

    @Test
    void outputFileCountMatchesValidInput() throws IOException {
        BatchConverter converter = new BatchConverter();
        BatchConverter.BatchResult result = converter.convert(getFixtureDir(), tempDir);

        // Count output files
        long outputFiles = Files.walk(tempDir)
                .filter(p -> p.toString().endsWith(".txt"))
                .count();
        assertEquals(result.succeeded(), (int) outputFiles);
    }
}
