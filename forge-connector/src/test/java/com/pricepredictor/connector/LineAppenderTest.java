package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LineAppenderTest {

    @Test
    void createsAMissingParentDirectoryAndAppendsEachLine(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("not").resolve("yet").resolve("lines.txt");
        LineAppender appender = new LineAppender(file);

        appender.append("first");
        appender.append("second");

        assertEquals(List.of("first", "second"), Files.readAllLines(file));
    }

    @Test
    void appendsToWhatIsAlreadyThere(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("lines.txt");
        Files.writeString(file, "old" + System.lineSeparator());

        new LineAppender(file).append("new");

        assertEquals(List.of("old", "new"), Files.readAllLines(file));
    }

    @Test
    void aMatchResultWriterNeedsNoExistingDirectory(@TempDir Path tmp) {
        Path file = tmp.resolve("draft").resolve("draft-games.txt");

        new MatchResultWriter(file);

        assertTrue(Files.isDirectory(file.getParent()));
    }
}
