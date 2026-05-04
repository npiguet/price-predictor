package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CardsPlayedWriter}'s open-write-close-per-line
 * strategy. Mirrors {@link MatchResultWriterTest} since both writers
 * follow the same concurrency-safe pattern.
 */
class CardsPlayedWriterTest {

    private static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    private static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    private static CardsPlayedRow row(String tag) {
        return new CardsPlayedRow(
                TS, RUN_ID, "RVR", "forge-best", "gen-" + tag,
                List.of("X"), List.of("Y"), List.of(), List.of(),
                'A', 'B');
    }

    @Test
    void writesOneLinePerCall(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("cards-played.txt");
        CardsPlayedWriter writer = new CardsPlayedWriter(file);

        writer.write(row("1"));
        writer.write(row("2"));
        writer.write(row("3"));

        List<String> lines = Files.readAllLines(file);
        assertEquals(3, lines.size());
    }

    @Test
    void appendsToExistingFile(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("cards-played.txt");
        Files.writeString(file, "previous-line\n");

        new CardsPlayedWriter(file).write(row("after"));

        List<String> lines = Files.readAllLines(file);
        assertEquals(2, lines.size());
        assertEquals("previous-line", lines.get(0));
    }

    @Test
    void concurrentWritersDoNotCorruptLines(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("cards-played.txt");
        CardsPlayedWriter writer = new CardsPlayedWriter(file);

        int writers = 4;
        int rowsPerWriter = 50;
        ExecutorService pool = Executors.newFixedThreadPool(writers);
        for (int t = 0; t < writers; t++) {
            final int id = t;
            pool.submit(() -> {
                for (int i = 0; i < rowsPerWriter; i++) {
                    writer.write(row(id + "-" + i));
                }
            });
        }
        pool.shutdown();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));

        List<String> lines = Files.readAllLines(file);
        assertEquals(writers * rowsPerWriter, lines.size());
        // Every line must have exactly the expected number of semicolons.
        for (String line : lines) {
            int count = (int) line.chars().filter(c -> c == ';').count();
            assertEquals(10, count, "concurrent writes corrupted a line: " + line);
        }
    }

    @Test
    void parentDirectoryAutoCreated(@TempDir Path tmp) throws IOException {
        Path nested = tmp.resolve("output").resolve("sealed").resolve("cards-played.txt");
        new CardsPlayedWriter(nested).write(row("1"));
        assertTrue(Files.exists(nested));
        assertEquals(1, Files.readAllLines(nested).size());
    }
}
