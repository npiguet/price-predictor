package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Converted text is a corpus: it must not depend on the machine's locale.
 * Under a Turkish default, {@code "FIGHT".toLowerCase()} gives a dotless
 * {@code ı}, so every case conversion in the connector names
 * {@code Locale.ROOT}, as {@link HoldoutCards} explains for card names.
 */
class CaseConversionLocaleTest {

    @Test
    void everyCaseConversionNamesItsLocale() throws IOException {
        List<Path> offenders;
        try (Stream<Path> files = Files.walk(Path.of("src/main/java"))) {
            offenders = files.filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> {
                        String text = read(path);
                        return text.contains("toLowerCase()") || text.contains("toUpperCase()");
                    })
                    .toList();
        }
        assertEquals(List.of(), offenders);
    }

    private static String read(Path path) {
        try {
            return Files.readString(path);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
