package com.pricepredictor.connector;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

/**
 * How every line-per-record input the workers read is split into records:
 * seat tables, generated-decks files, validation matches, holdout and
 * castability card lists, and {@code DeckBuilderMain}'s stdin.
 *
 * <p>One rule for all of them — each line trimmed, blank ones skipped — so a
 * trailing space or a final blank line a Python writer or an editor leaves
 * means the same thing to every reader. Readers that pair records by
 * position (validation outcomes with their matches) depend on the skipped
 * lines being the same everywhere they count.
 */
final class CorpusFiles {

    private CorpusFiles() {
    }

    /** The file's lines, UTF-8, each trimmed, blank ones dropped. */
    static List<String> nonBlankLines(Path file) throws IOException {
        return nonBlank(Files.readAllLines(file).stream()).toList();
    }

    /**
     * The reader's lines, each trimmed, blank ones dropped — lazily, so a
     * line-at-a-time protocol answers each line as it arrives rather than
     * waiting for the end of its input.
     */
    static Stream<String> nonBlankLines(BufferedReader reader) {
        return nonBlank(reader.lines());
    }

    private static Stream<String> nonBlank(Stream<String> lines) {
        return lines.map(String::trim).filter(line -> !line.isEmpty());
    }
}
