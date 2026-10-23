package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LineAppenderTest {

    @Test
    void createsAMissingParentDirectoryUpFrontAndAppendsEachLine(@TempDir Path tmp)
            throws IOException {
        Path file = tmp.resolve("not").resolve("yet").resolve("lines.txt");
        LineAppender appender = new LineAppender(file);
        assertTrue(Files.isDirectory(file.getParent()), "created before any append");

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

    /** Workers share one file: every line arrives, and none is interleaved with another. */
    @Test
    void concurrentAppendersKeepEveryLineWhole(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("lines.txt");
        LineAppender appender = new LineAppender(file);
        int writers = 4;
        int linesPerWriter = 50;
        Set<String> expected = new HashSet<>();
        ExecutorService pool = Executors.newFixedThreadPool(writers);
        for (int t = 0; t < writers; t++) {
            for (int i = 0; i < linesPerWriter; i++) {
                expected.add(line(t, i));
            }
            final int id = t;
            pool.submit(() -> {
                for (int i = 0; i < linesPerWriter; i++) {
                    appender.append(line(id, i));
                }
            });
        }
        pool.shutdown();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));

        List<String> lines = Files.readAllLines(file);
        assertEquals(writers * linesPerWriter, lines.size());
        assertEquals(expected, new HashSet<>(lines), "concurrent appends corrupted a line");
    }

    /** A line with the corpus files' separators in it. */
    private static String line(int writer, int index) {
        return writer + ";" + index + ";" + "card|".repeat(20) + "end";
    }
}
