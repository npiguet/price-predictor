package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.SourceTree;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.RandomAccessFile;
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
        // Forge's parser fails on it with no message; the warning still names
        // what was thrown rather than printing "null".
        assertTrue(result.warnings().get(0).matches("\\[malformed_card] \\w+(Error|Exception)"),
                result.warnings().get(0));
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

    // ── a card's text and sidecar are written whole, together ───────────

    /** Forge's own script, from the golden tree. */
    private static final Path BOLT = GOLDEN.resolve("cardsfolder/l/lightning_bolt.txt");

    private Path sourceWithBolt() throws IOException {
        Path source = tempDir.resolve("source");
        Files.createDirectories(source.resolve("l"));
        Files.copy(BOLT, source.resolve("l/lightning_bolt.txt"));
        return source;
    }

    private static List<Path> filesUnder(Path root) throws IOException {
        try (Stream<Path> files = Files.walk(root)) {
            return files.filter(Files::isRegularFile).toList();
        }
    }

    /** The malformed fixture's text: the parser makes no card of it. */
    @Test
    void aScriptThatFailsLeavesNoFileForItself() throws IOException {
        Path source = sourceWithBolt();
        Files.createDirectories(source.resolve("m"));
        Files.copy(FIXTURE.resolve("m/malformed_card.txt"), source.resolve("m/malformed_card.txt"));
        Path output = tempDir.resolve("output");

        BatchConverter.BatchResult result =
                new BatchConverter().convert(source, output, SourceTree.CARDSFOLDER);

        assertEquals(List.of("malformed_card"), result.failedScripts());
        assertEquals(
                Set.of(output.resolve("l/lightning_bolt.txt"),
                        output.resolve("l/lightning_bolt.provenance.json")),
                Set.copyOf(filesUnder(output)));
    }

    /**
     * The sidecar fails after the text is built and renamed into place (its
     * target is a directory that cannot be replaced): the text goes too, and
     * no temporary file is left behind.
     */
    @Test
    void aSidecarThatCannotBeWrittenTakesItsTextWithIt() throws IOException {
        Path source = sourceWithBolt();
        Path output = tempDir.resolve("output");
        Path sidecar = output.resolve("l/lightning_bolt.provenance.json");
        Files.createDirectories(sidecar.resolve("occupied"));

        BatchConverter.BatchResult result =
                new BatchConverter().convert(source, output, SourceTree.CARDSFOLDER);

        assertEquals(List.of("lightning_bolt"), result.failedScripts());
        assertTrue(result.warnings().get(0).contains(sidecar.toString()), result.warnings().toString());
        assertEquals(List.of(), filesUnder(output));
    }

    /**
     * Windows will not rename onto a file another handle holds open: the
     * rewrite of a pair whose sidecar is held fails naming the sidecar, and
     * the old pair stays whole, its text restored after it was replaced.
     */
    @Test
    @EnabledOnOs(OS.WINDOWS)
    void aRewriteOntoAHeldSidecarFailsNamingItAndKeepsTheOldPair() throws IOException {
        Path source = sourceWithBolt();
        Path output = tempDir.resolve("output");
        BatchConverter converter = new BatchConverter();
        converter.convert(source, output, SourceTree.CARDSFOLDER);
        Path text = output.resolve("l/lightning_bolt.txt");
        Path sidecar = output.resolve("l/lightning_bolt.provenance.json");
        Files.writeString(text, "old text");
        Files.writeString(sidecar, "old sidecar");

        BatchConverter.BatchResult result;
        try (RandomAccessFile held = new RandomAccessFile(sidecar.toFile(), "r")) {
            result = converter.convert(source, output, SourceTree.CARDSFOLDER);
            assertEquals("old sidecar".length(), held.length());
        }

        assertEquals(List.of("lightning_bolt"), result.failedScripts());
        assertTrue(result.warnings().get(0).contains(sidecar.toString()), result.warnings().toString());
        assertEquals("old text", Files.readString(text));
        assertEquals("old sidecar", Files.readString(sidecar));
        assertEquals(Set.of(text, sidecar), Set.copyOf(filesUnder(output)));
    }
}
