package com.pricepredictor.connector;

import java.io.UncheckedIOException;
import java.nio.file.Path;

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
 * exists to bound could never end (see {@code MatchWorkerMain#recordMatch}).
 *
 * <p>Appending is a {@link LineAppender}'s, so concurrent workers can all
 * append to the one shared progress file without corrupting a line.
 */
public class ProgressWriter {

    private final LineAppender appender;

    public ProgressWriter(Path outputFile) {
        this.appender = new LineAppender(outputFile);
    }

    /**
     * Append one line marking a completed match.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write() {
        appender.append("1");
    }
}
