package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ConvertMain#work}: the exit a failed script, an occupied tree and an
 * unsafe output path end on, and what {@code --replace} sweeps.
 */
@ExtendWith(ForgeExtension.class)
class ConvertMainTest {

    /** Forge's own script, from the golden tree. */
    private static final Path BOLT = Path.of("src/test/resources/golden-024/cardsfolder/l/lightning_bolt.txt");

    @TempDir
    Path tempDir;

    private Path cards;
    private Path output;

    private Path sourceWithBolt() throws IOException {
        cards = tempDir.resolve("forge/cardsfolder");
        Files.createDirectories(cards.resolve("l"));
        Files.copy(BOLT, cards.resolve("l/lightning_bolt.txt"));
        output = tempDir.resolve("converted/cardsfolder");
        return cards;
    }

    private String[] args(String... extra) {
        List<String> all = new ArrayList<>(List.of(
                "--cards-path", cards.toString(),
                "--output-path", output.toString(),
                // No token source: the token tree is skipped.
                "--tokens-path", tempDir.resolve("forge/tokenscripts").toString(),
                "--tokens-output-path", tempDir.resolve("converted/tokenscripts").toString()));
        all.addAll(List.of(extra));
        return all.toArray(String[]::new);
    }

    /** The exit {@code work} requested, with what it printed on stderr. */
    private record Ended(ToolMain.Exit exit, String stderr) {
    }

    private Ended run(String... args) {
        PrintStream original = System.err;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        System.setErr(new PrintStream(bytes, true, StandardCharsets.UTF_8));
        try {
            ToolMain.Exit exit = assertThrows(ToolMain.Exit.class, () -> ConvertMain.work(args));
            return new Ended(exit, bytes.toString(StandardCharsets.UTF_8));
        } finally {
            System.setErr(original);
        }
    }

    @Test
    void aBatchWithAFailingScriptExitsNonZeroNamingItAndLeavesNoFileForIt() throws IOException {
        sourceWithBolt();
        Files.createDirectories(cards.resolve("m"));
        Files.writeString(cards.resolve("m/malformed_card.txt"), "This is not a valid card script\n");

        Ended ended = run(args());

        assertEquals(ExitCode.FAILURE, ended.exit().code());
        assertTrue(ended.stderr().contains("1 of 2"), ended.stderr());
        assertTrue(ended.stderr().contains("malformed_card"), ended.stderr());
        assertTrue(Files.isRegularFile(output.resolve("l/lightning_bolt.txt")));
        assertFalse(Files.exists(output.resolve("m/malformed_card.txt")));
        assertFalse(Files.exists(output.resolve("m/malformed_card.provenance.json")));
    }

    @Test
    void aCleanBatchReturns() throws Exception {
        sourceWithBolt();

        ConvertMain.work(args());

        assertTrue(Files.isRegularFile(output.resolve("l/lightning_bolt.txt")));
        assertTrue(Files.isRegularFile(output.resolve("l/lightning_bolt.provenance.json")));
    }

    @Test
    void aTreeHoldingConvertedFilesIsRefusedWithoutReplace() throws IOException {
        sourceWithBolt();
        Path old = output.resolve("o/old_card.txt");
        Files.createDirectories(old.getParent());
        Files.writeString(old, "old");

        Ended ended = run(args());

        assertEquals(ExitCode.BAD_CONFIG, ended.exit().code());
        assertTrue(ended.stderr().contains("--replace"), ended.stderr());
        assertEquals("old", Files.readString(old));
        assertFalse(Files.exists(output.resolve("l/lightning_bolt.txt")));
    }

    /**
     * The sweep takes every converted file out of the tree, leftovers of a
     * killed run included, and nothing else: not the run manifest a caller
     * keeps there, not a file beside the tree.
     */
    @Test
    void replaceSweepsTheConvertedFilesAndNothingElse() throws Exception {
        sourceWithBolt();
        Path stale = output.resolve("s/stale_card.txt");
        Files.createDirectories(stale.getParent());
        Files.writeString(stale, "stale");
        Files.writeString(output.resolve("s/stale_card.provenance.json"), "{}");
        Files.writeString(output.resolve("s/half_written.provenance.json.tmp"), "{");
        Path manifest = Files.writeString(output.resolve("run.json"), "{}");
        Path sibling = Files.writeString(output.resolveSibling("notes.txt"), "keep me");

        ConvertMain.work(args("--replace"));

        assertFalse(Files.exists(output.resolve("s")), "the emptied directory is removed");
        assertTrue(Files.isRegularFile(output.resolve("l/lightning_bolt.txt")));
        assertEquals("{}", Files.readString(manifest));
        assertArrayEquals("keep me".getBytes(StandardCharsets.UTF_8), Files.readAllBytes(sibling));
    }

    @Test
    void anOutputTreeHoldingTheWorkingDirectoryIsRefused() throws IOException {
        sourceWithBolt();
        output = Path.of("").toAbsolutePath().getParent();

        Ended ended = run(args("--replace"));

        assertEquals(ExitCode.BAD_CONFIG, ended.exit().code());
        assertTrue(ended.stderr().contains(output.toString()), ended.stderr());
    }

    @Test
    void anOutputTreeOverTheSourceIsRefused() throws IOException {
        sourceWithBolt();
        output = cards.getParent();

        Ended ended = run(args("--replace"));

        assertEquals(ExitCode.BAD_CONFIG, ended.exit().code());
        assertTrue(Files.isRegularFile(cards.resolve("l/lightning_bolt.txt")), "the source is untouched");
    }

    @Test
    void aRepositoryRootIsRefused() throws IOException {
        sourceWithBolt();
        output = tempDir.resolve("checkout");
        Files.createDirectories(output.resolve(".git"));

        Ended ended = run(args("--replace"));

        assertEquals(ExitCode.BAD_CONFIG, ended.exit().code());
        assertTrue(ended.stderr().contains("repository"), ended.stderr());
    }
}
