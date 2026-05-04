package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends one {@link CardsPlayedRow} per call to the shared output file.
 *
 * <p>Each call opens the file, writes one line, and closes it — so each
 * write is flushed at line granularity and concurrent workers can append
 * without corrupting a line. Mirrors {@link MatchResultWriter}'s pattern.
 *
 * <p>Default target: {@code output/sealed/cards-played.txt}.
 */
public class CardsPlayedWriter {

    private final Path outputFile;

    public CardsPlayedWriter(Path outputFile) {
        this.outputFile = outputFile;
        Path parent = outputFile.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                throw new UncheckedIOException(
                        "Failed to create parent directory for " + outputFile, e);
            }
        }
    }

    /**
     * Append one {@link CardsPlayedRow} as a single line.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(CardsPlayedRow row) {
        String line = row.toLine();
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "Failed to write cards-played row to " + outputFile, e);
        }
    }
}
