package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link PlayedCardCollector}'s static filter helpers. The
 * full event-bus integration is exercised by
 * {@link MatchGeneratorCardsPlayedIntegrationTest} (Forge-tagged).
 */
class PlayedCardCollectorTest {

    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "all conditions hold,                                  true,  true,  false, false, true",
            // Threaten / Mind Control: controller != owner.
            "a stolen card is dropped,                             false, false, false, false, true",
            // Tokens are not deck cards.
            "a token is dropped,                                   false, true,  true,  false, true",
            // FR-004a: basic lands never appear in cards-played.txt.
            "a basic land is dropped at observation time,          false, true,  false, true,  true",
            // Emblems / "X's effect" trackers (The Ring, Monarch), dungeons, copied spells,
            // schemes, planes, etc. have a GamePieceType other than CARD and must not enter
            // the played-cards log.
            "a non-deck game piece is dropped,                     false, true,  false, false, false",
    })
    void shouldRecordOnlyARealDeckCardTheOwnerControls(
            String ignoredCase, boolean expected, boolean controllerEqualsOwner,
            boolean isToken, boolean isBasicLand, boolean isRealDeckCard) {
        assertEquals(expected, PlayedCardCollector.shouldRecord(
                controllerEqualsOwner, isToken, isBasicLand, isRealDeckCard));
    }

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
