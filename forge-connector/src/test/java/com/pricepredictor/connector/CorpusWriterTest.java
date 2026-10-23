package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.cardsPlayedRow;
import static com.pricepredictor.connector.MatchFixtures.header;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A {@link CorpusWriter} appends each row's line. The lines are the rows'
 * own tests' ({@link CardsPlayedRowTest}, {@link MatchResultTest}), and
 * appending is {@link LineAppenderTest}'s.
 */
class CorpusWriterTest {

    private static CardsPlayedRow row(String tag) {
        return cardsPlayedRow(header("gen-" + tag));
    }

    @Test
    void writesEachRowsLineInOrder(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("cards-played.txt");
        CorpusWriter<CardsPlayedRow> writer = new CorpusWriter<>(file);

        writer.write(row("1"));
        writer.write(row("2"));
        writer.write(row("3"));

        assertEquals(List.of(row("1").toLine(), row("2").toLine(), row("3").toLine()),
                Files.readAllLines(file));
    }
}
