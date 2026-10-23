package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The outcomes file a validation worker writes: one row per input line, in
 * input order, which is how the supervisor pairs each outcome with its match.
 *
 * <p>Every match here names cards Forge does not know, so it is refused before
 * a game is played and fills its row with {@code 0;0}; a played match's row is
 * {@link ValidationMatchPlayerIntegrationTest}'s.
 */
@ExtendWith(ForgeExtension.class)
class ValidationMatchPlayerTest {

    /** A match whose decks cannot be built. */
    private static final String UNPLAYABLE = "No Such Card|Not A Card;Fake Card|Not One Either";

    @TempDir
    Path tmp;

    private final ValidationMatchPlayer player = new ValidationMatchPlayer(new GamePlayer(1));

    private List<String> outcomesAfter(List<String> matches, Path outcomes) throws IOException {
        Path matchesFile = tmp.resolve("matches.txt");
        Files.write(matchesFile, matches);
        player.processAll(matchesFile, outcomes);
        return Files.readAllLines(outcomes);
    }

    @Test
    void aMatchThatCannotBePlayedStillWritesItsRow() throws IOException {
        assertEquals(List.of("0;0", "0;0"),
                outcomesAfter(List.of(UNPLAYABLE, UNPLAYABLE), tmp.resolve("outcomes.txt")));
    }

    @Test
    void aLineWithNoDeckBStillWritesItsRow() throws IOException {
        assertEquals(List.of("0;0"),
                outcomesAfter(List.of("Lightning Bolt|Mountain"), tmp.resolve("outcomes.txt")));
    }

    @Test
    void crashRecoverySkipsCompletedMatches() throws IOException {
        Path outcomes = tmp.resolve("outcomes.txt");
        // Two matches already completed before the restart.
        Files.writeString(outcomes, "2;0\n1;2\n");

        assertEquals(List.of("2;0", "1;2", "0;0"),
                outcomesAfter(List.of(UNPLAYABLE, UNPLAYABLE, UNPLAYABLE), outcomes));
    }

    @Test
    void aCompletedFileIsLeftAsItIs() throws IOException {
        Path outcomes = tmp.resolve("outcomes.txt");
        Files.writeString(outcomes, "2;0\n1;2\n");

        assertEquals(List.of("2;0", "1;2"), outcomesAfter(List.of(UNPLAYABLE, UNPLAYABLE), outcomes));
    }

    @Test
    void crashRecoveryUnreadableFileThrowsInsteadOfReplayingMatches() throws IOException {
        // A directory exists but cannot be read as a file; reporting 0 would replay every match.
        Path matchesFile = tmp.resolve("matches.txt");
        Files.write(matchesFile, List.of(UNPLAYABLE));
        Path outcomes = Files.createDirectory(tmp.resolve("outcomes"));

        assertThrows(IOException.class, () -> player.processAll(matchesFile, outcomes));
    }
}
