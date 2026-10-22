package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The invariants {@link CardsPlayedRow#of} owns, against real Forge card types. */
@ExtendWith(ForgeExtension.class)
class CardsPlayedRowOfTest {

    private static List<String> deck(String... nonBasics) {
        List<String> deck = new ArrayList<>(List.of(nonBasics));
        deck.addAll(List.of("Mountain", "Mountain", "Snow-Covered Island", "Wastes"));
        return deck;
    }

    private static MatchResult match(List<String> deckA, List<String> deckB) {
        return new MatchResult(Instant.parse("2026-04-22T14:30:05Z"), "run", "RVR",
                "forge-best", "random", deckA, deckB, "A", "B", 1);
    }

    @Test
    void columnsAreDistinctDisjointAndFreeOfBasicLands() {
        MatchResult parent = match(
                deck("Lightning Bolt", "Lightning Bolt", "Shock"),
                deck("Counterspell", "Opt"));
        GamePlayer.GameOutcome game = new GamePlayer.GameOutcome(
                Side.A, Side.B,
                List.of("Lightning Bolt", "Lightning Bolt"), List.of("Opt"));

        CardsPlayedRow row = CardsPlayedRow.of(parent, game);

        assertEquals(List.of("Lightning Bolt"), row.cardsPlayedA());
        assertEquals(List.of("Shock"), row.cardsNotPlayedA());
        assertEquals(List.of("Opt"), row.cardsPlayedB());
        assertEquals(List.of("Counterspell"), row.cardsNotPlayedB());
        assertEquals(Side.A, row.winner());
        assertEquals(Side.B, row.starter());
        assertEquals("run", row.runId());
        assertEquals("forge-best", row.methodA());
    }

    @Test
    void aBasicLandReportedAsPlayedIsDropped() {
        MatchResult parent = match(deck("Shock"), deck("Opt"));
        CardsPlayedRow row = CardsPlayedRow.of(parent, new GamePlayer.GameOutcome(
                Side.A, Side.A, List.of("Mountain", "Shock"), List.of("Snow-Covered Island")));

        assertEquals(List.of("Shock"), row.cardsPlayedA());
        assertTrue(row.cardsPlayedB().isEmpty());
        assertEquals(List.of("Opt"), row.cardsNotPlayedB());
    }

    @Test
    void aSideThatPlayedNothingKeepsEveryNonBasicNotPlayed() {
        MatchResult parent = match(deck("Shock"), deck("Opt"));
        CardsPlayedRow row = CardsPlayedRow.of(parent,
                new GamePlayer.GameOutcome(Side.B, Side.A, List.of(), List.of()));

        assertTrue(row.cardsPlayedA().isEmpty());
        assertEquals(List.of("Shock"), row.cardsNotPlayedA());
        assertFalse(row.toLine().contains("Mountain"));
    }

    @Test
    void everyBasicLandIsRecognisedByForgesTypeRule() {
        for (String basic : List.of("Plains", "Island", "Swamp", "Mountain", "Forest",
                "Wastes", "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
                "Snow-Covered Mountain", "Snow-Covered Forest")) {
            assertTrue(BasicLands.isBasic(basic), basic);
        }
        assertFalse(BasicLands.isBasic("Lightning Bolt"));
        assertFalse(BasicLands.isBasic("Not A Card Name"));
    }
}
