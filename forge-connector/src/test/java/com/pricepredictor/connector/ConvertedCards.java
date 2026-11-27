package com.pricepredictor.connector;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Card scripts converted through one {@link RulesParser}, for the tests that
 * read what the parser makes of a script: an inline one, or a file of Forge's
 * card folder.
 */
final class ConvertedCards {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser converter = new RulesParser();

    /** The card the script {@code lines} describe. */
    MultiCard convert(String... lines) {
        return converter.parseScript(Arrays.asList(lines), null).card();
    }

    /** The card the script at {@code relativePath} in Forge's card folder describes. */
    MultiCard convertFromFile(String relativePath) {
        try {
            Path file = CARDS_FOLDER.resolve(relativePath);
            List<String> lines = Files.readAllLines(file);
            return converter.parseScript(lines, null).card();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The first face of the card the script at {@code relativePath} describes. */
    CardFace face(String relativePath) {
        return convertFromFile(relativePath).faces().get(0);
    }
}
