package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.SourceTree;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(ForgeExtension.class)
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

    // ── the converted text must not change (FR-004, SC-001) ─────────────

    private static final Path GOLDEN = Path.of("src/test/resources/golden-024");

    /**
     * The sidecar's chained script text is a different file; the converted
     * text is the corpus every other consumer reads, and it was captured under
     * {@code golden-024/expected} before the chain renderer existed. A single
     * byte of drift there is a reconversion of the whole corpus.
     */
    @Test
    void convertingTheGoldenScriptsReproducesEveryGoldenTextByteForByte() throws IOException {
        BatchConverter converter = new BatchConverter();
        int compared = 0;
        for (String tree : List.of(SourceTree.CARDSFOLDER, SourceTree.TOKENSCRIPTS)) {
            Path output = tempDir.resolve(tree);
            BatchConverter.BatchResult result =
                    converter.convert(GOLDEN.resolve(tree), output, tree);
            assertEquals(List.of(), result.warnings(), tree);

            Path expected = GOLDEN.resolve("expected").resolve(tree);
            try (Stream<Path> files = Files.walk(expected)) {
                for (Path golden : files.filter(p -> p.toString().endsWith(".txt")).toList()) {
                    Path written = output.resolve(expected.relativize(golden));
                    assertTrue(Files.exists(written), "not written: " + written);
                    assertArrayEquals(Files.readAllBytes(golden), Files.readAllBytes(written),
                            "converted text drifted from the golden: " + golden);
                    compared++;
                }
            }
        }
        assertTrue(compared >= 30, "only " + compared + " goldens compared");
    }

    /** A conversion run reports its undefined SVars rather than printing per card. */
    @Test
    void anUndefinedSVarInTheTreeIsReportedOnTheResult() throws IOException {
        BatchConverter.BatchResult result = new BatchConverter().convert(
                GOLDEN.resolve(SourceTree.CARDSFOLDER), tempDir, SourceTree.CARDSFOLDER);

        assertEquals(1, result.missingSVars().size(), result.missingSVars().toString());
        assertEquals("Undefined SVar", result.missingSVars().get(0).card());
        assertEquals("DBMissing", result.missingSVars().get(0).label());
    }

    /** Drained per tree: a second tree's report does not repeat the first's. */
    @Test
    void theReportIsDrainedBetweenTrees() throws IOException {
        BatchConverter converter = new BatchConverter();
        converter.convert(GOLDEN.resolve(SourceTree.CARDSFOLDER),
                tempDir.resolve("cards"), SourceTree.CARDSFOLDER);
        BatchConverter.BatchResult tokens = converter.convert(
                GOLDEN.resolve(SourceTree.TOKENSCRIPTS),
                tempDir.resolve("tokens"), SourceTree.TOKENSCRIPTS);

        assertTrue(tokens.missingSVars().isEmpty(), tokens.missingSVars().toString());
    }
}
