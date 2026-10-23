package com.pricepredictor.connector;

import java.io.UncheckedIOException;
import java.nio.file.Path;

/**
 * Appends one {@link CorpusRow} per call to a corpus file shared by
 * concurrent workers: {@code match-outcomes.txt}, {@code cards-played.txt},
 * or a draft-games file.
 *
 * <p>The line is the row's {@link CorpusRow#toLine()}; appending is a
 * {@link LineAppender}'s, one open-write-close per call, so concurrent
 * workers can all append without corrupting one another's lines.
 *
 * @param <R> the one kind of row the file holds, so a match row cannot be
 *            appended to the cards-played file
 */
public final class CorpusWriter<R extends CorpusRow> {

    private final LineAppender appender;

    /** Creates the file's parent directory if it is missing. */
    public CorpusWriter(Path outputFile) {
        this.appender = new LineAppender(outputFile);
    }

    /**
     * Append the row's line.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(R row) {
        appender.append(row.toLine());
    }
}
