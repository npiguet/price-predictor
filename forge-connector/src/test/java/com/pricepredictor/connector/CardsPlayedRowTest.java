package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.time.format.DateTimeFormatter;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.HEADER;
import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static com.pricepredictor.connector.MatchFixtures.TS;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CardsPlayedRow}'s line-format contract.
 *
 * <p>Schema (eleven fields, semicolon-separated, no trailing {@code ;}):
 * {@code timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter}
 */
class CardsPlayedRowTest {

    private static CardsPlayedRow sample(
            List<String> cardsPlayedA, List<String> cardsPlayedB,
            List<String> cardsNotPlayedA, List<String> cardsNotPlayedB,
            Side winner, Side starter) {
        return new CardsPlayedRow(
                HEADER,
                cardsPlayedA, cardsPlayedB, cardsNotPlayedA, cardsNotPlayedB,
                winner, starter);
    }

    @Test
    void aLineIsElevenFieldsInTheDocumentedOrder() {
        CardsPlayedRow row = sample(
                List.of("Lightning Bolt", "Tarmogoyf"),
                List.of("Counterspell"),
                List.of("Snapcaster Mage"),
                List.of("Brainstorm"),
                Side.A, Side.B);
        String line = row.toLine();
        assertFalse(line.endsWith(";"), "line must not end with ';'");
        String[] parts = line.split(";", -1);
        assertEquals(11, parts.length);
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
                Side.A, Side.B);
        String[] parts = row.toLine().split(";", -1);
        assertEquals("", parts[5]);
        assertEquals("", parts[6]);
        assertEquals("", parts[7]);
        assertEquals("", parts[8]);
    }
}
