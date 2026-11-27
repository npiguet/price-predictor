package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The lobby names {@link Side} maps. {@link PlayedCardCollector}'s event-bus
 * integration is exercised by {@link MatchGeneratorCardsPlayedIntegrationTest}
 * (Forge-tagged).
 */
class PlayedCardCollectorTest {

    @Test
    void mapsLobbyNameToSide() {
        assertEquals(Side.A, Side.ofLobbyName("p1"));
        assertEquals(Side.B, Side.ofLobbyName("p2"));
    }

    @Test
    void unknownLobbyNameThrows() {
        // Never a default to B: a card or a game credited to the wrong deck
        // is a corrupt row.
        assertThrows(IllegalArgumentException.class, () -> Side.ofLobbyName("nobody"));
    }
}
