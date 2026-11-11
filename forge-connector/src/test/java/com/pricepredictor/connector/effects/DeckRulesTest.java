package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A game's aim from its two decks' rules (FR-058d): a game is steered when
 * either deck carries a rule, and its records then carry both decks' rules,
 * de-duplicated and sorted, instead of the run's list.
 */
class DeckRulesTest {

    @Test
    void aGameIsSteeredWhenEitherDeckCarriesARule() {
        assertFalse(DeckRules.NONE.steered());
        assertFalse(DeckRules.of(null, null).steered());
        assertTrue(DeckRules.of("704.5f", null).steered());
        assertTrue(DeckRules.of(null, "702.19b").steered());
        assertTrue(DeckRules.of("704.5f", "702.19b").steered());
    }

    @Test
    void aSteeredGameIsAimedAtBothDecksRulesDeduplicatedAndSorted() {
        assertEquals(List.of("702.19b", "704.5f"),
                DeckRules.of("704.5f", "702.19b").aimedAt(Optional.of(List.of())));
        assertEquals(List.of("704.5f"),
                DeckRules.of("704.5f", "704.5f").aimedAt(Optional.of(List.of())));
        assertEquals(List.of("702.19b"),
                DeckRules.of(null, "702.19b").aimedAt(Optional.of(List.of("100.1"))),
                "the decks' rules replace the run's list");
        assertEquals(List.of("702.19b"), DeckRules.of(null, "702.19b").aimedAt(Optional.empty()));
    }

    @Test
    void aGameNoDeckSteersIsAimedAtTheRunsList() {
        assertEquals(List.of(), DeckRules.NONE.aimedAt(Optional.of(List.of())));
        assertEquals(List.of("100.1"), DeckRules.of(null, null).aimedAt(Optional.of(List.of("100.1"))));
        assertNull(DeckRules.NONE.aimedAt(Optional.empty()), "a match-outcomes game aims at nothing");
    }
}
