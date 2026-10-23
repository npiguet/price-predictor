package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** The validation worker's two line formats: {@link ValidationMatch} in, {@link ValidationOutcome} out. */
class ValidationLineTest {

    @Test
    void aMatchLineIsDeckAThenDeckB() {
        assertEquals(new ValidationMatch(List.of("Shock", "Mountain"), List.of("Opt")),
                ValidationMatch.parse("Shock|Mountain;Opt"));
    }

    @Test
    void aMatchLineWithNoDeckBIsRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> ValidationMatch.parse("Lightning Bolt|Mountain"));
    }

    @Test
    void anOutcomeIsTheWinCountsSemicolonSeparated() {
        assertEquals("4;3", new ValidationOutcome(4, 3).toLine());
    }

    @Test
    void theFillerCountsNoGames() {
        assertEquals("0;0", ValidationOutcome.NONE.toLine());
    }

    @Test
    void aPlayedMatchWritesEachSidesWins() {
        GamePlayer.PlayedMatch played = new GamePlayer.PlayedMatch(List.of(
                new GamePlayer.GameOutcome(Side.B, Side.A, List.of(), List.of()),
                new GamePlayer.GameOutcome(Side.A, Side.B, List.of(), List.of()),
                new GamePlayer.GameOutcome(Side.B, Side.A, List.of(), List.of())), 12, false);

        assertEquals(new ValidationOutcome(1, 2), ValidationOutcome.of(played));
    }

    @Test
    void aNegativeWinCountIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> new ValidationOutcome(-1, 0));
    }
}
