package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** A played validation match's row: deck A's wins, then deck B's. */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class ValidationMatchPlayerIntegrationTest {

    @TempDir
    Path tmp;

    private static String deck(String land, int lands, String spell) {
        List<String> names = new ArrayList<>(Collections.nCopies(lands, land));
        names.addAll(Collections.nCopies(40 - lands, spell));
        return CardNameList.format(names);
    }

    /**
     * Deck B is forty Islands and cannot win, so the row's order is fixed by
     * the result rather than by luck.
     */
    @Test
    void aPlayedMatchWritesDeckAsWinsThenDeckBs() throws IOException {
        Path matches = tmp.resolve("matches.txt");
        Files.write(matches, List.of(
                deck("Mountain", 17, "Raging Goblin") + ";" + deck("Island", 40, "Island")));
        Path outcomes = tmp.resolve("outcomes.txt");

        new ValidationMatchPlayer(new GamePlayer(1)).processAll(matches, outcomes);

        assertEquals(List.of("1;0"), Files.readAllLines(outcomes));
    }
}
