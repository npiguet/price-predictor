package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends one {@link MatchResult} per call to the shared output file.
 *
 * <p>The line is {@link MatchResult#toLine()}.
 *
 * <p>Each call opens the file, writes one line, and closes it — ensuring each write
 * is flushed and concurrent workers can all append without corruption.
 */
public class MatchResultWriter {

    private final Path outputFile;

    public MatchResultWriter(Path outputFile) {
        this.outputFile = outputFile;
    }

    /**
     * Append one match result line to the output file.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(MatchResult result) {
        String line = result.toLine();

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write match result to " + outputFile, e);
        }
    }
}
