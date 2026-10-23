package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.SourceTree;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(ForgeExtension.class)
class BatchConverterTest {

    @TempDir
    Path tempDir;

    /** Two valid cards, {@code t/test_bear} and {@code t/test_flyer}, and one malformed. */
    private static final Path FIXTURE = Path.of("src/test/resources/cardsfolder");

    /**
     * Every file is read, the malformed one warns without stopping the run,
     * and each valid card is written at its own path in the mirrored tree.
     */
    @Test
    void aTreeWithAMalformedCardConvertsTheRestAndWarnsOnce() throws IOException {
        BatchConverter.BatchResult result =
                new BatchConverter().convert(FIXTURE, tempDir, SourceTree.CARDSFOLDER);

        assertEquals(3, result.totalFiles());
        assertEquals(2, result.succeeded());
        assertEquals(1, result.warningCount(), result.warnings().toString());
        try (Stream<Path> files = Files.walk(tempDir)) {
            assertEquals(
                    Set.of(tempDir.resolve("t/test_bear.txt"), tempDir.resolve("t/test_flyer.txt")),
                    files.filter(p -> p.toString().endsWith(".txt")).collect(Collectors.toSet()));
        }
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
        for (SourceTree tree : List.of(SourceTree.CARDSFOLDER, SourceTree.TOKENSCRIPTS)) {
            Path output = tempDir.resolve(tree.wireName());
            BatchConverter.BatchResult result =
                    converter.convert(GOLDEN.resolve(tree.wireName()), output, tree);
            assertEquals(List.of(), result.warnings(), tree.wireName());

            Path expected = GOLDEN.resolve("expected").resolve(tree.wireName());
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
                GOLDEN.resolve(SourceTree.CARDSFOLDER.wireName()), tempDir, SourceTree.CARDSFOLDER);

        assertEquals(1, result.missingSVars().size(), result.missingSVars().toString());
        assertEquals("Undefined SVar", result.missingSVars().get(0).card());
        assertEquals("DBMissing", result.missingSVars().get(0).label());
    }

    /** Drained per tree: a second tree's report does not repeat the first's. */
    @Test
    void theReportIsDrainedBetweenTrees() throws IOException {
        BatchConverter converter = new BatchConverter();
        converter.convert(GOLDEN.resolve(SourceTree.CARDSFOLDER.wireName()),
                tempDir.resolve("cards"), SourceTree.CARDSFOLDER);
        BatchConverter.BatchResult tokens = converter.convert(
                GOLDEN.resolve(SourceTree.TOKENSCRIPTS.wireName()),
                tempDir.resolve("tokens"), SourceTree.TOKENSCRIPTS);

        assertTrue(tokens.missingSVars().isEmpty(), tokens.missingSVars().toString());
    }
}
