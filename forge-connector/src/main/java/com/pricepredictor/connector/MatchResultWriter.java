package com.pricepredictor.connector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.StringJoiner;

/**
 * Appends one {@link MatchResult} per call to the shared output file.
 *
 * <p>Format: {@code deckA_card1|deckA_card2|...|deckA_card40;deckB_card1|...|deckB_card40;winsA;winsB}
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
     * @throws IllegalArgumentException if wins do not sum to 2 or 3.
     * @throws UncheckedIOException if the file cannot be written.
     */
    public void write(MatchResult result) {
        int total = result.winsA() + result.winsB();
        if (total != 2 && total != 3) {
            throw new IllegalArgumentException(
                    "winsA + winsB must be 2 or 3, got " + result.winsA() + " + " + result.winsB());
        }

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

    private static String formatLine(MatchResult result) {
        return encodeDeck(result.deckA())
                + ";" + encodeDeck(result.deckB())
                + ";" + result.winsA()
                + ";" + result.winsB();
    }

    private static String encodeDeck(List<String> cards) {
        StringJoiner joiner = new StringJoiner("|");
        for (String card : cards) {
            joiner.add(card);
        }
        return joiner.toString();
    }
}
