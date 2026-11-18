package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What {@link ConvertedTrees} counts as converted, and how its sweep fails. */
class ConvertedTreesTest {

    @TempDir
    Path tree;

    private Path pair(String dir, String stem) throws IOException {
        Path text = tree.resolve(dir).resolve(stem + ".txt");
        Files.createDirectories(text.getParent());
        Files.writeString(text, "text");
        Files.writeString(text.resolveSibling(stem + ".provenance.json"), "{}");
        return text;
    }

    @Test
    void aTextWithoutItsSidecarIsNotConverted() throws IOException {
        Path report = Files.writeString(tree.resolve("check_report.txt"), "report");
        Path text = pair("a", "card");

        assertFalse(ConvertedTrees.isConverted(report));
        assertTrue(ConvertedTrees.isConverted(text));
        assertTrue(ConvertedTrees.isConverted(text.resolveSibling("card.provenance.json")));
        assertTrue(ConvertedTrees.isConverted(text.resolveSibling("card.txt.tmp")));
    }

    @Test
    void anUnpairedTextDoesNotMakeATreeOccupied() throws IOException {
        Files.writeString(tree.resolve("check_report.txt"), "report");

        ConvertedTrees.refuseOccupied(List.of(tree));

        pair("a", "card");
        assertThrows(ToolMain.Exit.class, () -> ConvertedTrees.refuseOccupied(List.of(tree)));
    }

    @Test
    void theSweepKeepsAnUnpairedTextAndRemovesEveryPair() throws IOException {
        Path report = Files.writeString(tree.resolve("check_report.txt"), "report");
        Path text = pair("a", "card");

        assertEquals(2, ConvertedTrees.sweep(tree));

        assertTrue(Files.isRegularFile(report));
        assertFalse(Files.exists(text.getParent()));
    }

    @Test
    void aSidecarThatCannotBeRemovedFailsTheSweepNamingItAndLeavesWholePairs() throws IOException {
        Path first = pair("a", "alpha");
        Path held = pair("b", "beta").resolveSibling("beta.provenance.json");
        Path last = pair("c", "gamma");

        IOException e = assertThrows(IOException.class, () -> ConvertedTrees.sweep(tree, path -> {
            if (path.equals(held)) {
                throw new IOException("held open");
            }
            Files.delete(path);
        }));

        assertTrue(e.getMessage().contains(held.toString()), e.getMessage());
        assertTrue(Files.exists(held));
        assertTrue(Files.exists(held.resolveSibling("beta.txt")), "the text stays beside its sidecar");
        for (Path text : List.of(first, last)) {
            boolean text_ = Files.exists(text);
            boolean sidecar = Files.exists(text.resolveSibling(
                    text.getFileName().toString().replace(".txt", ".provenance.json")));
            assertEquals(text_, sidecar, "pair whole or gone: " + text);
        }
    }
}
