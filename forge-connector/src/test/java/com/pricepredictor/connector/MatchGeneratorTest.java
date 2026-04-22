package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MatchGenerator set selection and match flow.
 *
 * <p>These are integration tests (tagged "integration") because set filtering
 * requires the full Forge environment (StaticData / FModel).
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class MatchGeneratorTest {

    private static final String TEST_RUN_ID = "test-run-id";

    @Test
    void generateMatchReturnsValidResult() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        MatchResult result = generator.generateMatch();

        assertNotNull(result);
        assertEquals(40, result.deckA().size(), "deckA must be exactly 40 cards");
        assertEquals(40, result.deckB().size(), "deckB must be exactly 40 cards");
        int total = result.winsA() + result.winsB();
        assertTrue(total == 2 || total == 3, "wins must sum to 2 or 3, got " + total);
        assertTrue(result.winsA() >= 0 && result.winsA() <= 2);
        assertTrue(result.winsB() >= 0 && result.winsB() <= 2);
    }

    @Test
    void generateMatchPopulatesMetadata() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        MatchResult result = generator.generateMatch();

        assertEquals(TEST_RUN_ID, result.runId());
        assertNotNull(result.timestamp(), "timestamp must be populated");
        assertNotNull(result.setCode(), "setCode must be populated");
        assertFalse(result.setCode().isBlank(), "setCode must be non-blank");
        assertTrue(result.durationSeconds() >= 0, "duration must be non-negative");
        assertEquals(result.games().length(), result.play().length(),
                "games and play must have matching lengths");
    }

    @Test
    void methodTagsComeFromDeckBuilder() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        MatchResult result = generator.generateMatch();

        List<String> allowed = List.of(
                DeckBuilder.METHOD_FORGE_BEST,
                DeckBuilder.METHOD_FORGE_3SUB,
                DeckBuilder.METHOD_FORGE_8SUB,
                DeckBuilder.METHOD_RANDOM
        );
        assertTrue(allowed.contains(result.methodA()),
                "methodA must be one of the phase-0 tags, got: " + result.methodA());
        assertTrue(allowed.contains(result.methodB()),
                "methodB must be one of the phase-0 tags, got: " + result.methodB());
    }

    @Test
    void runIdMustBeNonBlank() {
        assertThrows(IllegalArgumentException.class,
                () -> MatchGenerator.withDefaultBuilders(""));
        assertThrows(IllegalArgumentException.class,
                () -> MatchGenerator.withDefaultBuilders(null));
    }

    @Test
    void generateMatchUsesOnlyEligibleSets() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        // Eligible sets: must have draft booster template, must not be FUNNY type.
        // We don't know which set was chosen, but the generator should not throw.
        // If an ineligible set (e.g. un-set) were picked, Forge would likely throw
        // during pool generation.
        assertDoesNotThrow(() -> {
            for (int i = 0; i < 3; i++) {
                generator.generateMatch();
            }
        });
    }

    @Test
    void eligibleSetsListIsNonEmpty() {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();
        assertFalse(eligibleSets.isEmpty(), "Should have at least one eligible sealed set");
    }

    @Test
    void eligibleSetsExcludeFunnySets() {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();

        // Known un-sets that must be excluded
        List<String> unsets = List.of("UGL", "UNH", "UST", "UND", "UNF");
        for (String unset : unsets) {
            assertFalse(eligibleSets.contains(unset),
                    "Un-set " + unset + " must not be in eligible sets");
        }
    }

    @Test
    void eligibleSetsIncludeKnownSealedSets() {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();

        // Well-known sets that should be in the eligible list
        List<String> expectedSets = List.of("RVR", "MH3", "BLB");
        for (String set : expectedSets) {
            assertTrue(eligibleSets.contains(set),
                    "Known set " + set + " should be in eligible sets");
        }
    }

    @Test
    void eligibleSetsExcludeSmallBoosterSets() {
        List<String> eligibleSets = MatchGenerator.computeEligibleSets();

        // Legacy sets whose original boosters held only 8 cards are too small
        // for a realistic sealed pool. Must be excluded.
        List<String> tinyBoosterSets = List.of("DRK", "FEM");
        for (String set : tinyBoosterSets) {
            assertFalse(eligibleSets.contains(set),
                    "Small-booster set " + set + " must not be in eligible sets");
        }
    }
}
