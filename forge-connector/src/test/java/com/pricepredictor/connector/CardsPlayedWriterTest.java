package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.TS;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A {@link CardsPlayedWriter} appends each row's line. The line is {@link
 * CardsPlayedRowTest}'s, and appending is {@link LineAppenderTest}'s.
 */
class CardsPlayedWriterTest {

    private static CardsPlayedRow row(String tag) {
        return new CardsPlayedRow(
                TS, RUN_ID, "RVR", "forge-best", "gen-" + tag,
                List.of("X"), List.of("Y"), List.of(), List.of(),
                Side.A, Side.B);
    }

    @Test
    void writesEachRowsLineInOrder(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("cards-played.txt");
        CardsPlayedWriter writer = new CardsPlayedWriter(file);

        writer.write(row("1"));
        writer.write(row("2"));
        writer.write(row("3"));

        assertEquals(List.of(row("1").toLine(), row("2").toLine(), row("3").toLine()),
                Files.readAllLines(file));
    }
}
