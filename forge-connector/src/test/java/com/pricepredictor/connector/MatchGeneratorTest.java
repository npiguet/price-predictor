package com.pricepredictor.connector;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;
import java.util.Random;
import forge.item.PaperCard;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
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


    /** Forge-built decks on both sides, best-of-3, every eligible set. */
    static MatchGenerator sealedSelfPlay() {
        return MatchGenerator.builder(BoosterTemplates.eligibleSealedSets(), RUN_ID).build();
    }

    private static MatchGenerationResult sharedMatch;

    /**
     * One sealed self-play match, played once for every test here and in
     * {@link MatchGeneratorCardsPlayedIntegrationTest} that only reads one:
     * a real Forge match costs seconds, and these tests ask nothing of the
     * match that a second one would answer differently.
     */
    static synchronized MatchGenerationResult sharedMatch() {
        if (sharedMatch == null) {
            sharedMatch = sealedSelfPlay().generateMatch();
        }
        return sharedMatch;
    }

    private static MatchResult result;

    @BeforeAll
    static void playOneMatch() {
        result = sharedMatch().matchResult();
    }

    @Test
    void generateMatchReturnsValidResult() {
        assertEquals(40, result.deckA().size(), "deckA must be exactly 40 cards");
        assertEquals(40, result.deckB().size(), "deckB must be exactly 40 cards");
        // Best-of-3: two or three games, and no deck wins more than two.
        List<GameResult> games = result.games();
        assertTrue(games.size() == 2 || games.size() == 3, "games: " + games);
        assertTrue(games.stream().filter(g -> g.winner() == Side.A).count() <= 2, "games: " + games);
        assertTrue(games.stream().filter(g -> g.winner() == Side.B).count() <= 2, "games: " + games);
    }

    @Test
    void generateMatchPopulatesMetadata() {
        assertEquals(RUN_ID, result.header().runId());
        assertNotNull(result.header().timestamp(), "timestamp must be populated");
        assertNotNull(result.header().setCode(), "setCode must be populated");
        assertFalse(result.header().setCode().isBlank(), "setCode must be non-blank");
    }

    @Test
    void methodTagsComeFromDeckBuilder() {
        List<String> allowed = List.of("forge-best", "forge-3sub", "forge-8sub", "random");
        assertTrue(allowed.contains(result.header().methodA()),
                "methodA must be one of the phase-0 tags, got: " + result.header().methodA());
        assertTrue(allowed.contains(result.header().methodB()),
                "methodB must be one of the phase-0 tags, got: " + result.header().methodB());
    }

    @Test
    void theMatchWasPlayedInAnEligibleSet() {
        // An ineligible set (an un-set, say) would fail pool generation, but a
        // wrong pick that still generated would go unnoticed without this.
        assertTrue(BoosterTemplates.eligibleSealedSets().contains(result.header().setCode()),
                "set not eligible: " + result.header().setCode());
    }

    @Test
    void eligibleSetsExcludeFunnySets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

        // Known un-sets that must be excluded
        List<String> unsets = List.of("UGL", "UNH", "UST", "UND", "UNF");
        for (String unset : unsets) {
            assertFalse(eligibleSets.contains(unset),
                    "Un-set " + unset + " must not be in eligible sets");
        }
    }

    @Test
    void eligibleSetsIncludeKnownSealedSets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

        // Well-known sets that should be in the eligible list
        List<String> expectedSets = List.of("RVR", "MH3", "BLB");
        for (String set : expectedSets) {
            assertTrue(eligibleSets.contains(set),
                    "Known set " + set + " should be in eligible sets");
        }
    }

    @Test
    void eligibleSetsExcludeSmallBoosterSets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

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
        List<String> reference = new PoolGenerator().generate("RVR", 1, HoldoutCards.NONE).get(0);
        Set<String> excluded = Set.copyOf(reference.subList(0, 10));

        MatchGenerator generator = MatchGenerator.builder(List.of("RVR"), RUN_ID)
                .random(new Random(42)).excludedCards(HoldoutCards.of(excluded)).build();

        List<PaperCard> pool = generator.generatePool("RVR");

        assertTrue(pool.size() >= 70, "Depleted pool too small: " + pool.size());
        for (PaperCard card : pool) {
            assertFalse(excluded.contains(card.getName()),
                    "Excluded card reached a collected game: " + card.getName());
        }
    }

    @Test
    void decksOnlyAcceptsAMirrorInsteadOfReachingForge() {
        // What this guards against: a coverage round's live-card pool
        // shrinks monotonically as cards are satisfied (deck_weights in
        // collect_coverage.py drops every satisfied card), so a round with
        // exactly one card left samples 23 nonlands with replacement from a
        // single candidate -- every deck that round builds is byte-identical,
        // deterministically. Under decksOnly, pickDeckB's non-mirror search
        // then always comes up empty. Falling through to forgeBuilt("COVERAGE")
        // -- a sentinel set code that resolves against no real Forge edition --
        // would throw on every single match attempt, since decksOnly forces
        // every attempt down this branch: no match would ever complete.
        GeneratedDecksIndex.GeneratedDeck deck = new GeneratedDecksIndex.GeneratedDeck(
                "coverage", "COVERAGE", List.of("Llanowar Elves"));
        GeneratedDecksIndex mirrorOnly = new GeneratedDecksIndex(List.of(deck, deck));
        // Both sides point at the same index, which is what
        // CollectorSupervisor does (one decks file, both --side-*-decks
        // paths) and what decksOnly requires: a Forge-built side does not
        // fail a decks-only match, it quietly turns it back into sealed
        // self-play. pickDeckB never reads side A -- only pickDeckA does --
        // so this changes nothing about the branch under test, which is
        // called directly below with an explicit set code and deck-A card list.
        MatchGenerator.DeckSource decks = new MatchGenerator.DeckSource.FileSampled(mirrorOnly);
        MatchGenerator generator = MatchGenerator.builder(List.of("RVR"), RUN_ID)
                .sideA(decks).sideB(decks, 1).random(new Random(7)).decksOnly(true).build();

        MatchGenerator.DeckSelection b = assertDoesNotThrow(
                () -> generator.pickDeckB("COVERAGE", List.of("Llanowar Elves")),
                "pickDeckB must accept a mirror rather than reach Forge's "
                        + "set-based builder for a decksOnly round");

        assertEquals("COVERAGE", b.setCode());
        assertEquals(List.of("Llanowar Elves"), b.cardNames());
        assertNotNull(b.deck(), "the mirror deck must still materialize");
    }
}
