package com.pricepredictor.connector;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Arrays;
import java.util.List;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static org.junit.jupiter.api.Assertions.*;

/**
 * What one real sealed self-play match reports: its decks, its games and its
 * header.
 *
 * <p>Tagged {@code integration} because it plays a whole Forge match; what
 * the generator decides without playing one is tested in {@link
 * MatchGeneratorDeckSourcingTest} and {@link BoosterTemplatesTest}.
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
        List<String> allowed = Arrays.stream(BuildMethod.values()).map(BuildMethod::label).toList();
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
}
