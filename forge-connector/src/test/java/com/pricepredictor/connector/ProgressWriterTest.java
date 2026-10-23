package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A {@link ProgressWriter} appends one line per call, which the supervisor
 * counts. Appending is {@link LineAppenderTest}'s.
 */
class ProgressWriterTest {

    @Test
    void writesOneLinePerCall(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("run-id.progress.txt");
        ProgressWriter writer = new ProgressWriter(file);

        writer.write();
        writer.write();
        writer.write();

        assertEquals(List.of("1", "1", "1"), Files.readAllLines(file));
    }
}
