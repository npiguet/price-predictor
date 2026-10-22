package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends one line per call to a file shared by concurrent workers.
 *
 * <p>Each call opens the file, writes the line, and closes it, so every line
 * is flushed whole and workers appending to the same file do not corrupt one
 * another's lines. The file's parent directory is created once, up front.
 * Lines end with the platform's line separator, which every Python reader
 * handles through universal newlines.
 */
final class LineAppender {

    private final Path file;

    LineAppender(Path file) {
        this.file = file;
        Path parent = file.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                throw new UncheckedIOException(
                        "Failed to create parent directory for " + file, e);
            }
        }
    }

    /** @throws UncheckedIOException if the file cannot be written */
    void append(String line) {
        try (BufferedWriter writer = Files.newBufferedWriter(file,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to append a line to " + file, e);
        }
    }
}
