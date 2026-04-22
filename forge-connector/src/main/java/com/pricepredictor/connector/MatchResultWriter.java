package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.StringJoiner;

/**
 * Appends one {@link MatchResult} per call to the shared output file.
 *
 * <p>Format (10 semicolon-delimited fields):
 * {@code timestamp;run_id;set_code;method_A;method_B;deckA_cards;deckB_cards;games;play;duration_s}
 * where {@code deckA_cards} and {@code deckB_cards} are pipe-separated card names.
 *
 * <p>Each call opens the file, writes one line, and closes it — ensuring each write
 * is flushed and concurrent workers can all append without corruption.
 */
public class MatchResultWriter {

    private final Path outputFile;

    public MatchResultWriter(Path outputFile) {
        this.outputFile = outputFile;
    }

    /**
     * Append one match result line to the output file.
     *
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(MatchResult result) {
        String line = formatLine(result);

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write match result to " + outputFile, e);
        }
    }

    static String formatLine(MatchResult result) {
        return DateTimeFormatter.ISO_INSTANT.format(result.timestamp())
                + ";" + result.runId()
                + ";" + result.setCode()
                + ";" + result.methodA()
                + ";" + result.methodB()
                + ";" + encodeDeck(result.deckA())
                + ";" + encodeDeck(result.deckB())
                + ";" + result.games()
                + ";" + result.play()
                + ";" + result.durationSeconds();
    }

    private static String encodeDeck(List<String> cards) {
        StringJoiner joiner = new StringJoiner("|");
        for (String card : cards) {
            joiner.add(card);
        }
        return joiner.toString();
    }
}
