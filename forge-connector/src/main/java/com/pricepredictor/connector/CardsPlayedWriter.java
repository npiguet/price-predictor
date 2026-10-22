package com.pricepredictor.connector;

import java.io.UncheckedIOException;
import java.nio.file.Path;

/**
 * Appends one {@link CardsPlayedRow} per call to the shared output file.
 *
 * <p>Appending is a {@link LineAppender}'s, as for {@link MatchResultWriter}.
 *
 * <p>Default target: {@code output/sealed/cards-played.txt}.
 */
public class CardsPlayedWriter {

    private final LineAppender appender;

    public CardsPlayedWriter(Path outputFile) {
        this.appender = new LineAppender(outputFile);
    }

    /**
     * Append one {@link CardsPlayedRow} as a single line.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(CardsPlayedRow row) {
        appender.append(row.toLine());
    }
}
