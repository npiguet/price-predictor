package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.Set;

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

    // ── decksOnly ────────────────────────────────────────────────────────────

    @Test
    void decksOnlyAlwaysRollsTheFileForSideB() {
        // A coverage deck belongs to no set, so the Forge-method branch would
        // try to open a booster for a set code Forge has never heard of.
        // Both indices present: decksOnly is refused without them, because a
        // missing one turns the round back into sealed self-play in silence.
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, RUN_ID, emptyIndex(), emptyIndex(), 1,
                new Random(7), Set.of(), true);

        for (int i = 0; i < 100; i++) {
            assertTrue(generator.rollIsFileSample(),
                    "decks-only mode rolled a Forge method");
        }
    }

    @Test
    void withoutDecksOnlyTheRollStillMixes() {
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, RUN_ID, null, emptyIndex(), 4,
                new Random(7), Set.of(), false);

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
        // Before this fix existed at all, this configuration silently fell
        // through to forgeBuilt("COVERAGE"), which NPEs on a sentinel set
        // code; a clear exception here is strictly better than either that
        // NPE or a second silent fall-through reintroducing it.
        GeneratedDecksIndex sideB = new GeneratedDecksIndex(List.of(
                new GeneratedDeck("coverage", "OTHERSET", List.of("Forest"))));
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, RUN_ID, emptyIndex(), sideB, 1,
                new Random(7), Set.of(), true);

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> generator.pickDeckB("COVERAGE", List.of("Llanowar Elves")));
        assertTrue(ex.getMessage().contains("COVERAGE"),
                "exception should name the set that had no candidate");
    }

    // ── decksOnly needs both indices ──────────────────────────────────────

    @Test
    void decksOnlyRefusesANullSideBIndex() {
        // With sideBIndex null, rollIsFileSample() returns false for every
        // match and the whole round silently plays ordinary sealed self-play
        // -- a decks-only run that never touches its decks file, reporting
        // success the entire time.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> new MatchGenerator(ELIGIBLE, new DeckBuilder(), new GamePlayer(),
                        RUN_ID, emptyIndex(), null, 1, new Random(0), Set.of(), true));
        assertTrue(ex.getMessage().contains("decksOnly"));
    }

    @Test
    void decksOnlyRefusesANullSideAIndex() {
        // The same hole on side A, and with no exception downstream at all:
        // pickDeckA would build deck A from a fresh Forge pool of a random
        // eligible set, so half of every coverage match would be a sealed
        // deck the round never asked for.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> new MatchGenerator(ELIGIBLE, new DeckBuilder(), new GamePlayer(),
                        RUN_ID, null, emptyIndex(), 1, new Random(0), Set.of(), true));
        assertTrue(ex.getMessage().contains("decksOnly"));
    }

    @Test
    void bothIndicesPresentIsTheOnlyDecksOnlyConfiguration() {
        assertDoesNotThrow(() -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                emptyIndex(), emptyIndex(), 1, new Random(0), Set.of(), true));
    }

    @Test
    void withoutDecksOnlyANullSideAIndexIsStillOrdinary() {
        // `sealed match-outcomes --side-b-decks` is exactly this shape and
        // must keep working.
        assertDoesNotThrow(() -> new MatchGenerator(
                ELIGIBLE, new DeckBuilder(), new GamePlayer(), RUN_ID,
                null, emptyIndex(), 4, new Random(0), Set.of(), false));
    }

    // ── unresolved card names ─────────────────────────────────────────────
    //
    // materializeDeck itself needs Forge's card database, so the resolution
    // half lives in MatchGeneratorTest under -Pintegration. What it does with
    // the names that missed is pure policy and belongs here.

    @Test
    void nothingUnresolvedSaysNothing() {
        assertDoesNotThrow(() -> MatchGenerator.reportUnresolved(List.of(), 40, true));
        assertDoesNotThrow(() -> MatchGenerator.reportUnresolved(List.of(), 40, false));
    }

    @Test
    void aDecksOnlyDeckWithAnUnresolvableNameThrowsNamingIt() {
        // A coverage or variant deck's names come from output/cardsfolder/
        // and from VariantRegistry, not from Forge. A name Forge misses is
        // dropped, the deck materializes as 17 basics, a game plays, a
        // progress line is appended, records are written -- and the card the
        // round exists to reach never entered a game. For collect-variants
        // this can be the whole run: every deck basics, exit code 0, zero
        // variant records.
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> MatchGenerator.reportUnresolved(
                        List.of("Bolt Variant 0", "Bolt Variant 1"), 40, true));

        assertTrue(ex.getMessage().contains("Bolt Variant 0"),
                "the exception must name the cards that did not resolve, got: "
                        + ex.getMessage());
        assertTrue(ex.getMessage().contains("Bolt Variant 1"));
        assertTrue(ex.getMessage().contains("2 of 40"),
                "the exception must say how many of how many missed, got: "
                        + ex.getMessage());
    }

    @Test
    void anOrdinarySealedDeckOnlyLogsTheMiss() {
        // Sealed self-play's names came out of Forge in the first place, so a
        // miss there is a surprise rather than a broken round -- and killing
        // a long self-play run over one is worse than saying so.
        assertDoesNotThrow(() -> MatchGenerator.reportUnresolved(
                List.of("Whatever"), 40, false));
    }

    @Test
    void aLongUnresolvedListIsCappedRatherThanDumped() {
        List<String> many = new java.util.ArrayList<>();
        for (int i = 0; i < 40; i++) {
            many.add("Missing " + i);
        }

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> MatchGenerator.reportUnresolved(many, 40, true));

        assertTrue(ex.getMessage().contains("40 of 40"));
        assertFalse(ex.getMessage().contains("Missing 39"),
                "a 40-name dump is not a message anyone reads");
        assertTrue(ex.getMessage().contains("more"),
                "a capped list must say it was capped, got: " + ex.getMessage());
    }
}
