package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ProgressWriter}'s open-write-close-per-call strategy.
 * Mirrors {@link CardsPlayedWriterTest} since both writers follow the same
 * concurrency-safe pattern.
 */
class ProgressWriterTest {

    @Test
    void writesOneLinePerCall(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("run-id.progress.txt");
        ProgressWriter writer = new ProgressWriter(file);

        writer.write();
        writer.write();
        writer.write();

        List<String> lines = Files.readAllLines(file);
        assertEquals(3, lines.size());
    }

    @Test
    void appendsToExistingFile(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("run-id.progress.txt");
        Files.writeString(file, "1\n");

        new ProgressWriter(file).write();

        List<String> lines = Files.readAllLines(file);
        assertEquals(2, lines.size());
    }

    @Test
    void concurrentWritersDoNotLoseLines(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("run-id.progress.txt");
        ProgressWriter writer = new ProgressWriter(file);

        int writers = 4;
        int writesPerWriter = 50;
        ExecutorService pool = Executors.newFixedThreadPool(writers);
        for (int t = 0; t < writers; t++) {
            pool.submit(() -> {
                for (int i = 0; i < writesPerWriter; i++) {
                    writer.write();
                }
            });
        }
        pool.shutdown();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));

        List<String> lines = Files.readAllLines(file);
        assertEquals(writers * writesPerWriter, lines.size());
    }

    @Test
    void parentDirectoryAutoCreated(@TempDir Path tmp) throws IOException {
        Path nested = tmp.resolve("output").resolve("effects").resolve("run-id.progress.txt");
        new ProgressWriter(nested).write();
        assertTrue(Files.exists(nested));
        assertEquals(1, Files.readAllLines(nested).size());
    }
}
