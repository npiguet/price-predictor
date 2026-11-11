package com.pricepredictor.connector;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Finding the card scripts under a source tree. */
public final class ScriptFiles {

    private ScriptFiles() {
    }

    /**
     * Every {@code .txt} file under {@code root}, at any depth, that holds a
     * {@code Name:} line.
     *
     * <p>The {@code Name:} test is what keeps a stray text file out of Forge's
     * custom-cards directory, which the variant tree is staged into: Forge
     * reads every {@code .txt} there as a card, a file naming no card builds
     * rules with no face, and {@code StaticData} throws on it while building
     * the card database, so every Forge startup on the machine dies (FR-058b).
     */
    public static List<Path> under(Path root) throws IOException {
        try {
            return textFilesUnder(root).stream().filter(ScriptFiles::namesACard).toList();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    /**
     * Every {@code .txt} file under {@code root}, at any depth, script or not:
     * what the converter reads, so a script it cannot parse, a missing
     * {@code Name:} line among the causes, is a warning it reports rather than
     * a file it never sees.
     */
    public static List<Path> textFilesUnder(Path root) throws IOException {
        try (var paths = Files.walk(root)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".txt"))
                    .toList();
        }
    }

    /** Whether the script declares a card name; read leniently, as a script's encoding is not checked here. */
    private static boolean namesACard(Path script) {
        try {
            String text = new String(Files.readAllBytes(script), StandardCharsets.UTF_8);
            return ScriptHeader.of(text.lines().toList()).name() != null;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
