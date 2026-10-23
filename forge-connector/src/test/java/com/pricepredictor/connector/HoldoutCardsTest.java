package com.pricepredictor.connector;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** The held-out card list's file format and the name folding both readers share. */
class HoldoutCardsTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty("sealed.exclude.cards");
    }

    @Test
    void loadReadsOneNamePerLineSkippingBlanks(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("holdout.txt");
        Files.write(file, List.of("soul echo", "", "  llanowar elves  ", "   "));

        HoldoutCards holdout = HoldoutCards.load(file);

        assertEquals(2, holdout.size());
        assertTrue(holdout.contains("llanowar elves"));
    }

    @Test
    void namesMatchWhateverTheirCase() {
        // The list is written from the lowercased converted tree; Forge spells
        // the same card in title case.
        HoldoutCards holdout = HoldoutCards.of(List.of("soul echo"));

        assertTrue(holdout.contains("Soul Echo"));
        assertFalse(holdout.contains("Soul Warden"));
    }

    @Test
    void anUnsetPropertyIsAFullStrengthRun() {
        assertSame(HoldoutCards.NONE, HoldoutCards.fromSystemProperty());
    }

    @Test
    void anUnreadableListFailsRatherThanFallingBackToFullStrength(@TempDir Path dir) {
        System.setProperty("sealed.exclude.cards", dir.resolve("missing.txt").toString());

        assertThrows(IllegalStateException.class, HoldoutCards::fromSystemProperty);
    }
}
