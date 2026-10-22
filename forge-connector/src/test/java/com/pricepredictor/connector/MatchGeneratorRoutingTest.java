package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import com.pricepredictor.connector.MatchGenerator.DeckSource;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure unit tests for {@link MatchGenerator}'s routing logic — no Forge
 * dependency. Covers configuration validation and the
 * {@link MatchGenerator#rollIsFileSample()} distribution. The full
 * {@link MatchGenerator#generateMatch()} flow is covered by the
 * Forge-dependent integration tests in {@link MatchGeneratorTest}.
 */
class MatchGeneratorRoutingTest {

    private static final String RUN_ID = "test-run-id";
    private static final List<String> ELIGIBLE = List.of("MH3", "BLB", "RVR");

    private static DeckSource oneDeckFile() {
        return file(new GeneratedDecksIndex(List.of(
                new GeneratedDeck("test-label", "MH3", List.of("A")))));
    }

    private static DeckSource file(GeneratedDecksIndex index) {
        return new DeckSource.FileSampled(index);
    }

    private static MatchGenerator.Builder generator() {
        return MatchGenerator.builder(ELIGIBLE, RUN_ID).random(new Random(0));
    }

    // ── configuration validation ──────────────────────────────────────────────

    @Test
    void runIdMustBeNonBlank() {
        assertThrows(IllegalArgumentException.class,
                () -> MatchGenerator.builder(ELIGIBLE, "").build());
        assertThrows(IllegalArgumentException.class,
                () -> MatchGenerator.builder(ELIGIBLE, null).build());
    }

    @Test
    void eligibleSetsMustBeNonEmpty() {
        assertThrows(IllegalArgumentException.class,
                () -> MatchGenerator.builder(List.of(), RUN_ID).build());
    }

    @Test
    void sideBWeightMustBeAtLeastOneWhenSideBSamplesAFile() {
        assertThrows(IllegalArgumentException.class,
                () -> generator().sideB(oneDeckFile(), 0).build());
        assertThrows(IllegalArgumentException.class,
                () -> generator().sideB(oneDeckFile(), -1).build());
    }

    @Test
    void sideBWeightUnusedWhenSideBIsForgeBuilt() {
        // The weight is unused when side B is Forge-built: plain sealed
        // self-play (both sides Forge-built, weight 0) must work.
        assertDoesNotThrow(() -> generator().sideB(DeckSource.FORGE_BUILT, 0).build());
    }

    // ── rollIsFileSample distribution ────────────────────────────────────────

    @Test
    void rollIsFileSampleAlwaysFalseWhenSideBIsForgeBuilt() {
        MatchGenerator gen = generator().build();
        for (int i = 0; i < 1000; i++) {
            assertFalse(gen.rollIsFileSample());
        }
    }

    @Test
    void rollIsFileSampleApproximatesWeightFraction() {
        // Weight 4 → fraction 4/(10+4) = 4/14 ≈ 0.286
        MatchGenerator gen = generator()
                .sideB(oneDeckFile(), 4).random(new Random(123)).build();

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
        MatchGenerator gen = generator()
                .sideB(oneDeckFile(), 8).random(new Random(456)).build();

        int trials = 18_000;
        int hits = 0;
        for (int i = 0; i < trials; i++) {
            if (gen.rollIsFileSample()) hits++;
        }
        double frac = hits / (double) trials;
        assertTrue(frac >= 0.42 && frac <= 0.47,
                "Fraction should be ~8/18 (≈0.444), got " + frac);
    }

    // ── decksOnly ────────────────────────────────────────────────────────────

    @Test
    void decksOnlyAlwaysRollsTheFileForSideB() {
        // A coverage deck belongs to no set, so the Forge-method branch would
        // try to open a booster for a set code Forge has never heard of.
        // Both sides from a file: decksOnly is refused without them, because
        // a missing one turns the round back into sealed self-play in silence.
        MatchGenerator generator = generator()
                .sideA(oneDeckFile()).sideB(oneDeckFile(), 1)
                .random(new Random(7)).decksOnly(true).build();

        for (int i = 0; i < 100; i++) {
            assertTrue(generator.rollIsFileSample(),
                    "decks-only mode rolled a Forge method");
        }
    }

    @Test
    void withoutDecksOnlyTheRollStillMixes() {
        MatchGenerator generator = generator()
                .sideB(oneDeckFile(), 4).random(new Random(7)).build();

        boolean sawForgeMethod = false;
        for (int i = 0; i < 200; i++) {
            if (!generator.rollIsFileSample()) {
                sawForgeMethod = true;
                break;
            }
        }
        assertTrue(sawForgeMethod, "the ordinary roll never picked a Forge method");
    }

    // ── pickDeckB under decksOnly (the mirror-fallback fix) ────────────────────
    //
    // pickDeckB's happy path (a mirror is accepted) materializes a Forge Deck
    // via DeckSelection.fromFile, so it needs FModel initialized and lives in
    // MatchGeneratorTest under -Pintegration instead. The one case that never
    // reaches materialization -- decksOnly with literally no deck of deck A's
    // set anywhere in the side-B index -- is pure routing logic and belongs
    // here, called directly rather than via pickDeckA so no side needs Forge.

    @Test
    void decksOnlyThrowsRatherThanReachForgeWhenSideBHasNoDeckOfTheSet() {
        // Not reachable from CollectorSupervisor, which always points both
        // sides at the same file -- this is the defensive branch for a
        // side-B index misconfigured to hold no deck of deck A's set at all.
        // Falling through to forgeBuilt("COVERAGE") would find no booster
        // for a sentinel set code; a clear exception naming the set is
        // strictly better than that or a second silent fall-through.
        DeckSource sideB = file(new GeneratedDecksIndex(List.of(
                new GeneratedDeck("coverage", "OTHERSET", List.of("Forest")))));
        MatchGenerator generator = generator()
                .sideA(oneDeckFile()).sideB(sideB, 1)
                .random(new Random(7)).decksOnly(true).build();

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> generator.pickDeckB("COVERAGE", List.of("Llanowar Elves")));
        assertTrue(ex.getMessage().contains("COVERAGE"),
                "exception should name the set that had no candidate");
    }

    // ── decksOnly needs a file on both sides ──────────────────────────────

    @Test
    void decksOnlyRefusesAForgeBuiltSideB() {
        // With side B Forge-built, rollIsFileSample() returns false for every
        // match and the whole round silently plays ordinary sealed self-play
        // -- a decks-only run that never touches its decks file, reporting
        // success the entire time.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> generator().sideA(oneDeckFile()).decksOnly(true).build());
        assertTrue(ex.getMessage().contains("decksOnly"));
    }

    @Test
    void decksOnlyRefusesAForgeBuiltSideA() {
        // The same hole on side A, and with no exception downstream at all:
        // pickDeckA would build deck A from a fresh Forge pool of a random
        // eligible set, so half of every coverage match would be a sealed
        // deck the round never asked for.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> generator().sideB(oneDeckFile(), 1).decksOnly(true).build());
        assertTrue(ex.getMessage().contains("decksOnly"));
    }

    @Test
    void bothSidesFromAFileIsTheOnlyDecksOnlyConfiguration() {
        assertDoesNotThrow(() -> generator()
                .sideA(oneDeckFile()).sideB(oneDeckFile(), 1).decksOnly(true).build());
    }

    @Test
    void withoutDecksOnlyAForgeBuiltSideAIsStillOrdinary() {
        // `sealed match-outcomes --side-b-decks` is exactly this shape and
        // must keep working.
        assertDoesNotThrow(() -> generator().sideB(oneDeckFile(), 4).build());
    }
}
