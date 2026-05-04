package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link PlayedCardCollector}'s static filter helpers. The
 * full event-bus integration is exercised by
 * {@link MatchGeneratorCardsPlayedIntegrationTest} (Forge-tagged).
 */
class PlayedCardCollectorTest {

    @Test
    void shouldRecordWhenAllConditionsHold() {
        assertTrue(PlayedCardCollector.shouldRecord(
                /*controllerEqualsOwner*/ true,
                /*isToken*/ false,
                /*isBasicLand*/ false));
    }

    @Test
    void dropsStolenCards() {
        // Threaten / Mind Control: controller != owner.
        assertFalse(PlayedCardCollector.shouldRecord(
                /*controllerEqualsOwner*/ false,
                /*isToken*/ false,
                /*isBasicLand*/ false));
    }

    @Test
    void dropsTokens() {
        // Tokens are not deck cards.
        assertFalse(PlayedCardCollector.shouldRecord(
                /*controllerEqualsOwner*/ true,
                /*isToken*/ true,
                /*isBasicLand*/ false));
    }

    @Test
    void dropsBasicLandsAtObservationTime() {
        // FR-004a: basic lands never appear in cards-played.txt.
        assertFalse(PlayedCardCollector.shouldRecord(
                /*controllerEqualsOwner*/ true,
                /*isToken*/ false,
                /*isBasicLand*/ true));
    }

    @Test
    void mapsLobbyNameToSide() {
        // GamePlayer constants: p1 → A, p2 → B.
        assertEquals('A', PlayedCardCollector.lobbyNameToSide("p1"));
        assertEquals('B', PlayedCardCollector.lobbyNameToSide("p2"));
    }

    @Test
    void unknownLobbyNameThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> PlayedCardCollector.lobbyNameToSide("nobody"));
    }

    @Test
    void getCardsReturnsEmptyListForUnknownSide() {
        PlayedCardCollector collector = new PlayedCardCollector();
        assertTrue(collector.getCards('A').isEmpty());
        assertTrue(collector.getCards('B').isEmpty());
    }
}
