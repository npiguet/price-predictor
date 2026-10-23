package com.pricepredictor.connector;

import java.io.UncheckedIOException;
import java.nio.file.Path;

/**
 * Appends one {@link MatchResult} per call to the shared output file.
 *
 * <p>The line is {@link MatchResult#toLine()}.
 *
 * <p>Appending is a {@link LineAppender}'s: one open-write-close per call, so
 * concurrent workers can all append without corruption.
 */
public class MatchResultWriter {

    private final LineAppender appender;

    /** Creates the file's parent directory if it is missing. */
    public MatchResultWriter(Path outputFile) {
        this.appender = new LineAppender(outputFile);
    }

    /**
     * Append one match result line to the output file.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(MatchResult result) {
        appender.append(result.toLine());
    }
}
