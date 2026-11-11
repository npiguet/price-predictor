package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Which files under a tree are card scripts (FR-058b, python R1).
 *
 * <p>Forge reads every {@code .txt} staged into its custom-cards directory as a
 * card, and a {@code .txt} without a {@code Name:} line builds rules with no
 * face, on which {@code StaticData} throws while building the whole card
 * database. So a {@code .txt} is a script only when it names a card.
 */
class ScriptFilesTest {

    @Test
    void onlyATxtFileHoldingANameLineIsAScript(@TempDir Path tree) throws IOException {
        Files.writeString(tree.resolve("x_variant_0.txt"),
                "Name:X Variant 0\nManaCost:R\nTypes:Instant\n");
        // Gen-2's loadable list: card names, one per line, no Name: line.
        Files.writeString(tree.resolve("loadable.txt"), "X Variant 0\nY Variant 1\n");
        Files.writeString(tree.resolve("empty.txt"), "");
        Files.writeString(tree.resolve("x_variant_0.provenance.json"), "Name:Not A Script\n");
        Path nested = Files.createDirectories(tree.resolve("n"));
        Files.writeString(nested.resolve("nested_card.txt"),
                "  Name:Nested Card\nTypes:Land\n");

        List<String> found = ScriptFiles.under(tree).stream()
                .map(path -> tree.relativize(path).toString().replace('\\', '/'))
                .sorted()
                .toList();

        assertEquals(List.of("n/nested_card.txt", "x_variant_0.txt"), found);
    }
}
