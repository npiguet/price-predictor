package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends one line per completed match to a scratch progress file.
 *
 * <p>The Python supervisor counts this file's lines to know when a bounded
 * round has played its match budget ({@code ForgeWorkerPool.output_line_count}
 * / {@code CollectorSupervisor.progress_path}). The line's content carries no
 * meaning — it is a tally, not a corpus — so a match writes it unconditionally,
 * independent of records-only mode and of both sealed writers
 * ({@link MatchResultWriter}, {@link CardsPlayedWriter}): a counter must not
 * ride the same gate that keeps a coverage or variant round out of
 * {@code match-outcomes.txt} / {@code cards-played.txt}, or the round it
 * exists to bound could never end (that was Task 4's original defect — see
 * {@code MatchWorkerMain#recordMatch}).
 *
 * <p>Mirrors {@link CardsPlayedWriter}'s open-write-close-per-call pattern —
 * each call opens the file, writes one line, and closes it, so writes are
 * flushed at line granularity and concurrent workers can all append to the
 * one shared progress file without corrupting a line.
 */
public class ProgressWriter {

    private final Path outputFile;

    public ProgressWriter(Path outputFile) {
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
     * Append one line marking a completed match.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write() {
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write("1");
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "Failed to write progress line to " + outputFile, e);
        }
    }
}
