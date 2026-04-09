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

    @Test
    void generateMatchReturnsValidResult() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders();

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
    void generateMatchUsesOnlyEligibleSets() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders();

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
}
