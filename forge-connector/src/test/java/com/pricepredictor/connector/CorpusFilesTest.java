package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The one line-splitting rule every line-per-record input is read under. */
class CorpusFilesTest {

    private static final String PADDED = "  Lightning Bolt|Shock \r\n\n   \n\tÉowyn, Lady of Rohan\n";

    @Test
    void aFileReadsAsItsTrimmedNonBlankLines(@TempDir Path tmp) throws IOException {
        Path file = Files.writeString(tmp.resolve("lines.txt"), PADDED, StandardCharsets.UTF_8);

        assertEquals(List.of("Lightning Bolt|Shock", "Éowyn, Lady of Rohan"),
                CorpusFiles.nonBlankLines(file));
    }

    @Test
    void aReaderReadsUnderTheSameRule() {
        BufferedReader reader = new BufferedReader(new StringReader(PADDED));

        assertEquals(List.of("Lightning Bolt|Shock", "Éowyn, Lady of Rohan"),
                CorpusFiles.nonBlankLines(reader).toList());
    }

    @Test
    void anAllBlankFileHasNoLines(@TempDir Path tmp) throws IOException {
        Path file = Files.writeString(tmp.resolve("blank.txt"), "\n  \n");

        assertEquals(List.of(), CorpusFiles.nonBlankLines(file));
    }
}
