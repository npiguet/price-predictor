package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;
import java.util.Random;
import forge.item.PaperCard;

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

        MatchResult result = generator.generateMatch().matchResult();

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

        MatchResult result = generator.generateMatch().matchResult();

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

        MatchResult result = generator.generateMatch().matchResult();

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

    @Test
    void generatePoolOmitsExcludedCardsInTheCollectionPath() {
        // The pool every collected effect record is played from. `generate-pools
        // --exclude-cards` never reaches here: MatchGenerator makes its own pool
        // per match, so a depleted collection run needs the exclusion on the
        // worker, not on the pool file.
        List<String> reference = new PoolGenerator().generate("RVR", 1).get(0);
        Set<String> excluded = Set.copyOf(reference.subList(0, 10));

        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, TEST_RUN_ID, null, null, 0,
                new Random(42), excluded, false);

        List<PaperCard> pool = generator.generatePool("RVR");

        assertFalse(pool.isEmpty(), "Depleted pool came back empty");
        for (PaperCard card : pool) {
            assertFalse(excluded.contains(card.getName()),
                    "Excluded card reached a collected game: " + card.getName());
        }
    }

    @Test
    void generatePoolWithoutExclusionsIsFullStrength() {
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, TEST_RUN_ID, null, null, 0,
                new Random(42), Set.of(), false);

        assertTrue(generator.generatePool("RVR").size() >= 70,
                "Full-strength pool too small");
    }

    @Test
    void decksOnlyAcceptsAMirrorInsteadOfReachingForge() {
        // The bug this guards against: a coverage round's live-card pool
        // shrinks monotonically as cards are satisfied (deck_weights in
        // collect_coverage.py drops every satisfied card), so a round with
        // exactly one card left samples 23 nonlands with replacement from a
        // single candidate -- every deck that round builds is byte-identical,
        // deterministically. Under decksOnly, pickDeckB's non-mirror search
        // then always comes up empty. Before this fix, it fell through to
        // forgeBuilt("COVERAGE") -- a sentinel set code that resolves against
        // no real Forge edition -- throwing NullPointerException on every
        // single match attempt, forever, since decksOnly forces every attempt
        // down this branch: no match ever completed, the progress file never
        // grew, and ForgeWorkerPool.run() blocked indefinitely.
        GeneratedDecksIndex.GeneratedDeck deck = new GeneratedDecksIndex.GeneratedDeck(
                "coverage", "COVERAGE", List.of("Llanowar Elves"));
        GeneratedDecksIndex mirrorOnly = new GeneratedDecksIndex(List.of(deck, deck));
        // Both sides point at the same index, which is what
        // CollectorSupervisor does (one decks file, both --side-*-decks
        // paths) and what decksOnly now requires: a null index on either
        // side does not fail a decks-only match, it quietly turns it back
        // into sealed self-play. pickDeckB never reads sideAIndex -- only
        // pickDeckA does -- so this changes nothing about the branch under
        // test, which is called directly below with an explicit set code and
        // deck-A card list.
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, TEST_RUN_ID, mirrorOnly, mirrorOnly, 1,
                new Random(7), Set.of(), true);

        MatchGenerator.DeckSelection b = assertDoesNotThrow(
                () -> generator.pickDeckB("COVERAGE", List.of("Llanowar Elves")),
                "pickDeckB must accept a mirror rather than reach Forge's "
                        + "set-based builder for a decksOnly round");

        assertEquals("COVERAGE", b.setCode());
        assertEquals(List.of("Llanowar Elves"), b.cardNames());
        assertNotNull(b.deck(), "the mirror deck must still materialize");
    }
}
