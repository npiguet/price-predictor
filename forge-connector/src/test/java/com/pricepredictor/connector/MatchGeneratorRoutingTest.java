package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure unit tests for {@link MatchGenerator}'s routing logic — no Forge
 * dependency. Covers constructor validation and the
 * {@link MatchGenerator#rollIsFileSample()} distribution. The full
 * {@link MatchGenerator#generateMatch()} flow is covered by the
 * Forge-dependent integration tests in {@link MatchGeneratorTest}.
 */
class MatchGeneratorRoutingTest {

    private static final String RUN_ID = "test-run-id";
    private static final List<String> ELIGIBLE = List.of("MH3", "BLB", "RVR");

    private static GeneratedDecksIndex emptyIndex() {
        return new GeneratedDecksIndex(List.of(
                new GeneratedDeck("test-label", "MH3", List.of("A"))));
    }

    // ── constructor validation ────────────────────────────────────────────────

    @Test
    void runIdMustBeNonBlank() {
        assertThrows(IllegalArgumentException.class, () -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), "",
                null, null, 0, new Random(0)));
        assertThrows(IllegalArgumentException.class, () -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), null,
                null, null, 0, new Random(0)));
    }

    @Test
    void eligibleSetsMustBeNonEmpty() {
        assertThrows(IllegalArgumentException.class, () -> new MatchGenerator(
                List.of(), new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, null, 0, new Random(0)));
    }

    @Test
    void sideBWeightMustBeAtLeastOneWhenSideBIndexProvided() {
        GeneratedDecksIndex sideB = emptyIndex();
        assertThrows(IllegalArgumentException.class, () -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, sideB, 0, new Random(0)));
        assertThrows(IllegalArgumentException.class, () -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, sideB, -1, new Random(0)));
    }

    @Test
    void sideBWeightUnusedWhenSideBIndexNull() {
        // Passing weight=0 (or anything) is fine when sideBIndex is null —
        // the weight is unused. Phase-0 (both null, weight=0) must work.
        assertDoesNotThrow(() -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, null, 0, new Random(0)));
    }

    // ── rollIsFileSample distribution ────────────────────────────────────────

    @Test
    void rollIsFileSampleAlwaysFalseWhenSideBIndexNull() {
        MatchGenerator gen = new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, null, 0, new Random(0));
        for (int i = 0; i < 1000; i++) {
            assertFalse(gen.rollIsFileSample());
        }
    }

    @Test
    void rollIsFileSampleApproximatesWeightFraction() {
        // Weight 4 → fraction 4/(10+4) = 4/14 ≈ 0.286
        MatchGenerator gen = new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, emptyIndex(), 4, new Random(123));

        int trials = 14_000;
        int hits = 0;
        for (int i = 0; i < trials; i++) {
            if (gen.rollIsFileSample()) hits++;
        }
        double frac = hits / (double) trials;
        assertTrue(frac >= 0.255 && frac <= 0.315,
                "Fraction should be ~4/14 (≈0.286), got " + frac);
    }

    @Test
    void rollIsFileSampleScalesWithWeight() {
        // Weight 8 → fraction 8/(10+8) = 8/18 ≈ 0.444
        MatchGenerator gen = new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, emptyIndex(), 8, new Random(456));

        int trials = 18_000;
        int hits = 0;
        for (int i = 0; i < trials; i++) {
            if (gen.rollIsFileSample()) hits++;
        }
        double frac = hits / (double) trials;
        assertTrue(frac >= 0.42 && frac <= 0.47,
                "Fraction should be ~8/18 (≈0.444), got " + frac);
    }
}
