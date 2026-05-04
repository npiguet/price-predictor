package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CardsPlayedRow}'s line-format contract.
 *
 * <p>Schema (eleven fields, semicolon-separated, no trailing {@code ;}):
 * {@code timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter}
 */
class CardsPlayedRowTest {

    private static final Instant TS = Instant.parse("2026-04-22T14:30:05Z");
    private static final String RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab";

    private static CardsPlayedRow sample(
            List<String> cardsPlayedA, List<String> cardsPlayedB,
            List<String> cardsNotPlayedA, List<String> cardsNotPlayedB,
            char winner, char starter) {
        return new CardsPlayedRow(
                TS, RUN_ID, "RVR", "forge-best", "gen-2",
                cardsPlayedA, cardsPlayedB, cardsNotPlayedA, cardsNotPlayedB,
                winner, starter);
    }

    @Test
    void lineHasElevenSemicolonSeparatedFields() {
        CardsPlayedRow row = sample(
                List.of("Lightning Bolt"), List.of("Counterspell"),
                List.of("Mountain"), List.of("Island"),
                'A', 'B');
        String[] parts = row.toLine().split(";", -1);
        assertEquals(11, parts.length);
    }

    @Test
    void noTrailingSemicolon() {
        CardsPlayedRow row = sample(
                List.of("X"), List.of("Y"), List.of(), List.of(), 'A', 'A');
        String line = row.toLine();
        assertFalse(line.endsWith(";"), "line must not end with ';'");
    }

    @Test
    void fieldsAppearInExpectedOrder() {
        CardsPlayedRow row = sample(
                List.of("Lightning Bolt", "Tarmogoyf"),
                List.of("Counterspell"),
                List.of("Snapcaster Mage"),
                List.of("Brainstorm"),
                'A', 'B');
        String[] parts = row.toLine().split(";", -1);
        assertEquals(DateTimeFormatter.ISO_INSTANT.format(TS), parts[0]);
        assertEquals(RUN_ID, parts[1]);
        assertEquals("RVR", parts[2]);
        assertEquals("forge-best", parts[3]);
        assertEquals("gen-2", parts[4]);
        assertEquals("Lightning Bolt|Tarmogoyf", parts[5]);
        assertEquals("Counterspell", parts[6]);
        assertEquals("Snapcaster Mage", parts[7]);
        assertEquals("Brainstorm", parts[8]);
        assertEquals("A", parts[9]);
        assertEquals("B", parts[10]);
    }

    @Test
    void emptyCardListsRoundTripAsEmptyString() {
        CardsPlayedRow row = sample(
                List.of(), List.of(), List.of(), List.of(),
                'A', 'B');
        String[] parts = row.toLine().split(";", -1);
        assertEquals("", parts[5]);
        assertEquals("", parts[6]);
        assertEquals("", parts[7]);
        assertEquals("", parts[8]);
    }

    @Test
    void cardListPreservesIterationOrder() {
        // Row writer emits names in the order the input list provides; it does
        // not deduplicate (that's the caller's job — see MatchGenerator).
        CardsPlayedRow row = sample(
                List.of("Lightning Bolt", "Counterspell", "Tarmogoyf"),
                List.of("Brainstorm"),
                List.of(), List.of(),
                'A', 'A');
        String[] parts = row.toLine().split(";", -1);
        assertEquals("Lightning Bolt|Counterspell|Tarmogoyf", parts[5]);
    }

    @Test
    void timestampUsesIsoInstantFormat() {
        CardsPlayedRow row = sample(
                List.of(), List.of(), List.of(), List.of(),
                'A', 'B');
        String[] parts = row.toLine().split(";", -1);
        // Should round-trip through ISO_INSTANT parser
        assertEquals(TS, Instant.from(DateTimeFormatter.ISO_INSTANT.parse(parts[0])));
    }

    @Test
    void invalidWinnerCharRejected() {
        assertThrows(IllegalArgumentException.class, () -> sample(
                List.of(), List.of(), List.of(), List.of(),
                'X', 'A'));
    }

    @Test
    void invalidStarterCharRejected() {
        assertThrows(IllegalArgumentException.class, () -> sample(
                List.of(), List.of(), List.of(), List.of(),
                'A', 'C'));
    }
}
