package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Finding the card scripts under a source tree. */
public final class ScriptFiles {

    private ScriptFiles() {
    }

    /** Every {@code .txt} file under {@code root}, at any depth. */
    public static List<Path> under(Path root) throws IOException {
        try (var paths = Files.walk(root)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".txt"))
                    .toList();
        }
    }
}
